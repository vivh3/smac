from smac.env.starcraft2.starcraft2 import StarCraft2Env
from smac.env.starcraft2.starcraft2 import races, difficulties, Direction
from smac.env.starcraft2.starcraft2 import actions as actions_api
from operator import attrgetter
from copy import deepcopy
import numpy as np

from absl import logging

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol, run_parallel, portspicker

from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb


class StarCraft2EnvMulti(StarCraft2Env):
    def __init__(
        self,
        heuristic_team_1=False,
        heuristic_team_2=False,
        *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_reward_team_2 = (self.n_agents * self.reward_death_value + self.reward_win)
        self.last_action = np.zeros((self.n_agents + self.n_enemies, self.n_actions))
        self.heuristic_team_1 = heuristic_team_1
        self.heuristic_team_2 = heuristic_team_2
        self.action_error = 0
        self.games_won_team_1 = 0 # allies
        self.games_won_team_2 = 0 # enemies

    def _launch_game(self):
        # References: https://github.com/deepmind/pysc2/blob/master/pysc2/env/sc2_env.py
        #             as well as starcraft2.py

        self._run_config = run_configs.get(version=self.game_version)
        self.parallel = run_parallel.RunParallel() # Needed for multiplayer
        _map = maps.get(self.map_name)
        interface_options = sc_pb.InterfaceOptions(raw=True, score=False)

        num_players = 2

        if num_players > 1:
            # Reserve a whole bunch of ports for multiplayer implementation
            _ports = portspicker.pick_unused_ports(num_players * 2)
            logging.debug("Ports used for multiplayer: %s", _ports)
        else:
            _ports = []

        # Actually launch the game processes.
        self._sc2_proc = [
            self._run_config.start(extra_ports=_ports,
                                   window_size=self.window_size,
                                   want_rgb=False)
            for _ in range(num_players)]
        self._controller = [p.controller for p in self._sc2_proc]

        # Request to create the game
        create = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(
                map_path=_map.path,
                map_data=self._run_config.map_data(_map.path)),
                realtime=False,
                random_seed=self._seed)
        # Save the maps so they can access it.
        for c in self._controller:
            c.save_map(_map.path, _map.data(self._run_config))
        for _ in range(num_players):
            create.player_setup.add(type=sc_pb.Participant)
        self._controller[0].create_game(create)
        
        # Create the join requests.
        # For "agent" team's race
        join_reqs = []
        join = sc_pb.RequestJoinGame(race=races[self._agent_race],
                                     options=interface_options)
        join.shared_port = 0 # unused
        join.server_ports.game_port = _ports[0]
        join.server_ports.base_port = _ports[1]
        for i in range(num_players - 1):
            join.client_ports.add(game_port=_ports[i * 2 + 2],
                                  base_port=_ports[i * 2 + 3])
        join_reqs.append(join)
        # For "bot" team's race
        join = sc_pb.RequestJoinGame(race=races[self._bot_race],
                                     options=interface_options)
        join.shared_port = 0 # unused
        join.server_ports.game_port = _ports[0]
        join.server_ports.base_port = _ports[1]
        for i in range(num_players - 1):
            join.client_ports.add(game_port=_ports[i * 2 + 2],
                                  base_port=_ports[i * 2 + 3])
        join_reqs.append(join)

        # Join the game. This must be run in parallel because Join is a blocking
        # call to the game that waits until all clients have joined.
        self.parallel.run((c.join_game, join__) for c, join__ in
                          zip(self._controller, join_reqs))

        game_info = self._controller[0].game_info()
        map_info = game_info.start_raw
        map_play_area_min = map_info.playable_area.p0
        map_play_area_max = map_info.playable_area.p1
        self.max_distance_x = map_play_area_max.x - map_play_area_min.x
        self.max_distance_y = map_play_area_max.y - map_play_area_min.y
        self.map_x = map_info.map_size.x
        self.map_y = map_info.map_size.y

        if map_info.pathing_grid.bits_per_pixel == 1:
            vals = np.array(list(map_info.pathing_grid.data)).reshape(
                self.map_x, int(self.map_y / 8))
            self.pathing_grid = np.transpose(np.array([
                [(b >> i) & 1 for b in row for i in range(7, -1, -1)]
                for row in vals], dtype=np.bool))
        else:
            self.pathing_grid = np.invert(np.flip(np.transpose(np.array(
                list(map_info.pathing_grid.data), dtype=np.bool).reshape(
                    self.map_x, self.map_y)), axis=1))

        self.terrain_height = np.flip(
            np.transpose(np.array(list(map_info.terrain_height.data))
                         .reshape(self.map_x, self.map_y)), 1) / 255

    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self._episode_steps = 0
        if self._episode_count == 0:
            # Launch StarCraft II
            self._launch_game()
        else:
            self._restart()

        # Information kept for counting the reward
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.win_counted = False
        self.defeat_counted = False

        self.last_action = np.zeros((self.n_agents + self.n_enemies, self.n_actions))

        try:
            self._obs = []
            for c in self._controller:
                self._obs.append(c.observe())
            self.init_units()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

        if self.debug:
            logging.debug("Started Episode {}"
                          .format(self._episode_count).center(60, "*"))

        # Create storage for team variables 
        self.dist_traveled_team_1 = [0 for _ in range(self.n_agents)]
        self.dist_traveled_team_2 = [0 for _ in range(self.n_enemies)]
        self.prev_end_rel_pos_team_1 = [(al_unit.pos.x, al_unit.pos.y) for 
                                al_id, al_unit in self.agents.items()]
        self.prev_end_rel_pos_team_2 = [(e_unit.pos.x, e_unit.pos.y) for
                                e_id, e_unit in self.enemies.items()]
        self.stop_actions_team_1 = [0 for _ in range(self.n_agents)]
        self.stop_actions_team_2 = [0 for _ in range(self.n_enemies)]
        self.move_actions_team_1 = [0 for _ in range(self.n_agents)]
        self.move_actions_team_2 = [0 for _ in range(self.n_enemies)]
        self.attack_actions_team_1 = [0 for _ in range(self.n_agents)]
        self.attack_actions_team_2 = [0 for _ in range(self.n_enemies)]

        return self.get_obs(), self.get_state()

    def _restart(self):
        """Restart the environment by killing all units on the map.
        There is a trigger in the SC2Map file, which restarts the
        episode when there are no units left.
        """
        try:
            self._kill_all_units()
            for _ in range(3):
                for c in self._controller:
                    c.step()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

    def full_restart(self):
        """Full restart. Closes the SC2 process and launches a new one. """
        for p in self._sc2_proc:
            p.close()
        try:
            self._launch_game()
            self.force_restarts += 1
        except:
            self.full_restart()

    def step(self, actions):
        actions_int = [int(a) for a in actions]

        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        # not using heuristics by default during initialization
        if self.heuristic_team_1:
            for i in range(self.n_agents):
                actions_int[i] = self.get_heuristic_action(i)
        if self.heuristic_team_2:
            for i in range(self.n_enemies):
                actions_int[self.n_agents + i] = self.get_heuristic_action(self.n_agents + i)

        # Store the count of each action type for every unit (separately for the two teams)
        # For allies
        for i in range(self.n_agents):
            if actions_int[i] == 1: # stop
                self.stop_actions_team_1[i] += 1
            elif actions_int[i] > 1: # move north, southeast, west
                self.move_actions_team_1[i] += 1
            elif actions_int[i] > 5: # attack enemy (or heal if medivac)
                self.attack_actions_team_1[i] += 1
        # For enemies
        for i in range(self.n_enemies):
            if actions_int[self.n_agents + i] == 1:
                self.stop_actions_team_2[i] += 1
            elif actions_int[self.n_agents + i] > 1:
                self.move_actions_team_2[i] += 1
            elif actions_int[self.n_agents + i] > 5:
                self.attack_actions_team_2[i] += 1 

        # Collect individual actions
        sc_actions_team_1 = []
        sc_actions_team_2 = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))
        try:
            for a_id, action in enumerate(actions_int):
                agent_action = self.get_agent_action(a_id, action)
                if agent_action:
                    if a_id < self.n_agents: # part of allies/team 1
                        sc_actions_team_1.append(agent_action)
                    else: # part of enemies/team 2
                        sc_actions_team_2.append(agent_action)
        except AssertionError as err:
            self._episode_count += 1
            self.action_error += 1
            self.reset()
            # no reward, terminated true, info
            return [0 for _ in actions_int], True, {"game_won_team_1": False,
                                                    "game_won_team_2": False,
                                                    "env_error": True}

        # Send action request
        req_actions_p1 = sc_pb.RequestAction(actions=sc_actions_team_1)
        req_actions_p2 = sc_pb.RequestAction(actions=sc_actions_team_2)
        req_actions_all = [req_actions_p1, req_actions_p2]
        try:
            for idx, (controller, req_actions) in enumerate(zip(self._controller, req_actions_all)):
                controller.actions(req_actions)
                # Make step in SC2, i.e. apply actions
            if self._step_mul is not None:
                for _ in range(self._step_mul):
                    for c in self._controller:
                        c.step()
            # Observe here so that we know if the episode is over.
            for idx, c in enumerate(self._controller):
                self._obs[idx] = c.observe()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            # no reward, terminated true, info
            return [0 for _ in actions_int], True, {"game_won_team_1": False,
                                                    "game_won_team_2": False,
                                                    "env_error": True}
        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        game_end_code = self.update_units()

        terminated = False
        reward = self.reward_battle()
        info = {"game_won_team_1": False,
                "game_won_team_2": False}

        # Update teams' new position lists
        new_end_rel_pos_team_1 = []
        new_end_rel_pos_team_2 = []
        for i in range(self.n_agents):
            al_unit = self.get_unit_by_id(i)
            new_end_rel_pos_team_1.append( (al_unit.pos.x, al_unit.pos.y) )
        for i in range(self.n_enemies):
            e_unit = self.get_unit_by_id(self.n_agents + i)
            new_end_rel_pos_team_2.append( (e_unit.pos.x, e_unit.pos.y) )

        # Update teams' distance traveled lists
        for i in range(len(self.dist_traveled_team_1)):
            self.dist_traveled_team_1[i] += self.distance(
                self.prev_end_rel_pos_team_1[i][0], self.prev_end_rel_pos_team_1[i][1],
                new_end_rel_pos_team_1[i][0], new_end_rel_pos_team_1[i][1])
        for i in range(len(self.dist_traveled_team_2)):
            self.dist_traveled_team_2[i] += self.distance(
                self.prev_end_rel_pos_team_2[i][0], self.prev_end_rel_pos_team_2[i][1],
                new_end_rel_pos_team_2[i][0], new_end_rel_pos_team_2[i][1])

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1

            # store teams' relative positions at end of game
            center_x = self.map_x / 2
            center_y = self.map_y / 2
            end_rel_pos_team_1 = []
            end_rel_pos_team_2 = []
            for i in range(self.n_agents):
                a_unit = self.get_unit_by_id(i)
                end_rel_pos_team_1.append(((a_unit.pos.x - center_x) / self.max_distance_x,
                                           (a_unit.pos.y - center_y) / self.max_distance_y))
            for i in range(self.n_enemies):
                e_unit = self.get_unit_by_id(self.n_agents + i)
                end_rel_pos_team_2.append(((e_unit.pos.x - center_x) / self.max_distance_x,
                                           (e_unit.pos.y - center_y) / self.max_distance_y))
            # Case 1: Team 1 won
            if game_end_code == 1 and not self.win_counted:
                self.games_won_team_1 += 1
                self.win_counted = True
                info["game_won_team_1"] = True
                if not self.reward_sparse:
                    reward[0] += self.reward_win
                    reward[1] += self.reward_defeat
                else:
                    reward[0] = 1
                    reward[1] = -1

            # Case 2: Team 2 won
            elif game_end_code == -1 and not self.defeat_counted:
                self.games_won_team_2 += 1
                self.defeat_counted = True
                info["game_won_team_2"] = True
                if not self.reward_sparse:
                    reward[0] += self.reward_defeat
                    reward[1] += self.reward_win
                else:
                    reward[0] = -1
                    reward[1] = 1

        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

            # Case 3: Draw
            center_x = self.map_x / 2
            center_y = self.map_y / 2
            end_rel_pos_team_1 = []
            end_rel_pos_team_2 = []
            for i in range(self.n_agents):
                a_unit = self.get_unit_by_id(i)
                end_rel_pos_team_1.append(((a_unit.pos.x - center_x) / self.max_distance_x,
                                           (a_unit.pos.y - center_y) / self.max_distance_y))
            for i in range(self.n_enemies):
                e_unit = self.get_unit_by_id(self.n_agents + i)
                end_rel_pos_team_2.append(((e_unit.pos.x - center_x) / self.max_distance_x,
                                           (e_unit.pos.y - center_y) / self.max_distance_y))

        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, '-'))

        if terminated:
            self._episode_count += 1

        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate

        # concatenate rewards for all units on both teams
        reward_total = []
        for _ in range(self.n_agents):
            reward_total.append(reward[0])
        for _ in range(self.n_enemies):
            reward_total.append(reward[1])

        return reward_total, terminated, info

    def get_agent_action(self, a_id, action):
        if action <= 5: # stop or move
            return super().get_agent_action(a_id, action)
        else:
            avail_actions = self.get_avail_agent_actions(a_id)
            assert avail_actions[action] == 1, \
                "Agent {} cannot perform action {}".format(a_id, action)

            unit = self.get_unit_by_id(a_id)
            tag = unit.tag

            ally = a_id < self.n_agents
            # attack/heal units that are in range
            if ally:
                target_id = action - self.n_actions_no_attack
                if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                    target_unit = self.agents[target_id]
                    action_name = "heal"
                else:
                    target_unit = self.enemies[target_id]
                    action_name = "attack"
            else:
                target_id = action - self.n_actions_no_attack
                if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                    target_unit = self.enemies[target_id]
                    action_name = "heal"
                else:
                    target_unit = self.agents[target_id]
                    action_name = "attack"

            action_id = actions_api[action_name]
            target_tag = target_unit.tag

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False)

            if self.debug:
                logging.debug("Agent {} {}s unit # {}".format(
                    a_id, action_name, target_id))

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action


    def get_heuristic_action(self, a_id):
        """ Returns the heuristic action (not an action in starcraft2.py)"""
        avail_agent_actions = self.get_avail_agent_actions(a_id)

        # For an ally unit (Team 1)
        if a_id < self.n_agents:
            al_unit = self.get_unit_by_id(a_id)
            own_pos_x = al_unit.pos.x
            own_pos_y = al_unit.pos.y
            # have ally attack closest enemy if attacking is a possible action
            if sum(avail_agent_actions[self.n_actions_no_attack: self.n_actions]) > 0 \
                    and al_unit.health > 0:
                # find closest enemy
                dist_from_enemies = []
                for e_id, e_unit in self.enemies.items():
                    if avail_agent_actions[self.n_actions_no_attack + e_id]:
                        dist_from_enemies.append(
                            self.distance(own_pos_x, own_pos_y, e_unit.pos.x, e_unit.pos.y))
                    else:
                        dist_from_enemies.append(float('Inf'))
                return self.n_actions_no_attack + dist_from_enemies.index(min(dist_from_enemies))
            else:
                # If no target in range, move right or stop
                if avail_agent_actions[4]:
                    return 4
                elif avail_agent_actions[1]:
                    return 1
                else:
                    return 0
        # For an enemy unit (Team 2)
        else:
            e_unit = self.get_unit_by_id(a_id)
            own_pos_x = e_unit.pos.x
            own_pos_y = e_unit.pos.y
            # have enemy attack closest ally if attacking is a possible action
            if sum(avail_agent_actions[self.n_actions_no_attack: self.n_actions]) > 0 \
                    and e_unit.health > 0:
                # find the closest
                dist_from_allies = []
                for al_id, al_unit in self.agents.items():
                    if avail_agent_actions[self.n_actions_no_attack + al_id]:
                        dist_from_allies.append(
                            self.distance(own_pos_x, own_pos_y, al_unit.pos.x, al_unit.pos.y))
                    else:
                        dist_from_allies.append(float('Inf'))
                return self.n_actions_no_attack + dist_from_allies.index(min(dist_from_allies))
            else:
                # If no target in range, move right or stop
                if avail_agent_actions[5]:
                    return 5
                elif avail_agent_actions[1]:
                    return 1
                else:
                    return 0

    def reward_battle(self):
        """Reward function when self.reward_spare==False.
        Returns accumulative hit/shield point damage dealt to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to ally units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        """
        if self.reward_sparse:
            return 0

        reward = []
        delta_deaths_ally = 0  
        delta_deaths_enemy = 0 
        delta_ally = 0  
        delta_enemy = 0 

        neg_scale = self.reward_negative_scale

        # update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:
                # did not die so far
                prev_health = (
                    self.previous_ally_units[al_id].health
                    + self.previous_ally_units[al_id].shield
                )
                if al_unit.health == 0:
                    # just died
                    self.death_tracker_ally[al_id] = 1
                    if not self.reward_only_positive:
                        delta_deaths_ally -= self.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale
                    delta_deaths_enemy += self.reward_death_value
                else:
                    # still alive
                    delta_ally += neg_scale * (
                        prev_health - al_unit.health - al_unit.shield
                    )

        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:
                # did not die so far
                prev_health = (
                    self.previous_enemy_units[e_id].health
                    + self.previous_enemy_units[e_id].shield
                )
                if e_unit.health == 0:
                    # just died
                    self.death_tracker_enemy[e_id] = 1
                    if not self.reward_only_positive:
                        delta_deaths_enemy -= self.reward_death_value * neg_scale
                    delta_enemy += prev_health * neg_scale
                    delta_deaths_ally += self.reward_death_value
                else:
                    # still alive
                    delta_enemy += neg_scale * (
                            prev_health - e_unit.health - e_unit.shield)

        if self.reward_only_positive:
            reward.append(abs(delta_enemy + delta_deaths_ally)) # shield regeneration
            reward.append(abs(delta_ally + delta_deaths_enemy))
        else:
            reward.append(delta_enemy + delta_deaths_ally - delta_ally)
            reward.append(delta_ally + delta_deaths_enemy - delta_enemy)

        return np.array(reward)

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id. The observation is composed of:

           - agent movement features (where it can move to, height information and pathing grid)
           - enemy features (available_to_attack, health, relative_x, relative_y, shield, unit_type)
           - ally features (visible, distance, relative_x, relative_y, shield, unit_type)
           - agent unit features (health, shield, unit_type)

           All of this information is flattened and concatenated into a list,
           in the aforementioned order. To know the sizes of each of the
           features inside the final list of features, take a look at the
           functions ``get_obs_move_feats_size()``,
           ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` and
           ``get_obs_own_feats_size()``.

           The size of the observation vector may vary, depending on the
           environment configuration and type of units present in the map.
           For instance, non-Protoss units will not have shields, movement
           features may or may not include terrain height and pathing grid,
           unit_type is not included if there is only one type of unit in the
           map etc.).
           NOTE: Agents should have access only to their local observations
           during decentralised execution.
           NOTE 2: enemy_feat represents enemies of agent_id, NOT self.enemies
        """
        unit = self.get_unit_by_id(agent_id)
        is_ally = agent_id < self.n_agents
        own_al_or_e_id = agent_id if is_ally else agent_id - self.n_agents

        # cannot use get_obs_own_feats_size bc own unit here could be
        # an ally or enemy 
        own_feats_dim = self.unit_type_bits
        if self.obs_own_health:
            if is_ally:
                own_feats_dim += 1 + self.shield_bits_ally
            else:
                own_feats_dim += 1 + self.shield_bits_enemy
        if self.obs_bool_team:
            own_feats_dim += 2 # for one hot encoding of the unit's "team id"
        if self.obs_own_position:
            own_feats_dim += 2 # for x, y

        move_feats_dim = self.get_obs_move_feats_size()
        enemy_feats_dim = self.get_obs_enemy_feats_size()
        ally_feats_dim = self.get_obs_ally_feats_size()

        own_feats = np.zeros(own_feats_dim, dtype=np.float32)
        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)

        if unit.health > 0:  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)

            # Movement features
            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]

            ind = self.n_actions_move

            if self.obs_pathing_grid:
                move_feats[
                    ind : ind + self.n_obs_pathing
                ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.obs_terrain_height:
                move_feats[ind:] = self.get_surrounding_height(unit)

            al_ids = [
                al_id for al_id in range(self.n_agents)
                if not is_ally or (is_ally and al_id != agent_id)
            ]
            en_ids = [
                self.n_agents + en_id for en_id in range(self.n_enemies)
                if is_ally or (not is_ally and en_id != own_al_or_e_id)
            ]
            if not is_ally:
                al_ids, en_ids = en_ids, al_ids

            # Enemy features
            for i, en_id in enumerate(en_ids): # en_id - i = self.n_agents
                e_unit = self.get_unit_by_id(en_id)
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if dist < sight_range and e_unit.health > 0: # visible and alive
                    # Sight range > shoot range
                    enemy_feats[i, 0] = avail_actions[
                        self.n_actions_no_attack + i
                    ]  # available
                    enemy_feats[i, 1] = dist / sight_range  # distance
                    enemy_feats[i, 2] = (e_x - x) / sight_range # relative X
                    enemy_feats[i, 3] = (e_y - y) / sight_range # relative Y

                    ind = 4
                    if self.obs_all_health:
                        enemy_feats[i, ind] = (
                            e_unit.health / e_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            enemy_feats[i, ind] = (
                                e_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(e_unit, False)
                        enemy_feats[i, ind + type_id] = 1  # unit type
                        ind += self.unit_type_bits

            # Ally features
            for i, al_id in enumerate(al_ids):
                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)

                if dist < sight_range and al_unit.health > 0:
                    # if visible and alive
                    ally_feats[i, 0] = 1
                    ally_feats[i, 1] = dist / sight_range  # distance
                    ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                    ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        ally_feats[i, ind] = (
                            al_unit.health / al_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            ally_feats[i, ind] = (
                                al_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True)
                        ally_feats[i, ind + type_id] = 1
                        ind += self.unit_type_bits

                    if self.obs_last_action:
                        ally_feats[i, ind:] = self.last_action[al_id]

            # Own features
            ind = 0
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1

            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit, True)
                own_feats[ind + type_id] = 1
                ind += 1

            if self.obs_bool_team:
                if is_ally:
                    own_feats[ind] = 1
                else:
                    own_feats[ind + 1] = 1
                ind += 2

            if self.obs_own_position:
                own_feats[ind] = (x - (self.map_x / 2)) / self.max_distance_x  # relative X
                own_feats[ind + 1] = (y - (self.map_y / 2)) / self.max_distance_y  # relative Y
                ind += 2

        agent_obs = np.concatenate(
            (
                move_feats.flatten(),
                enemy_feats.flatten(),
                ally_feats.flatten(),
                own_feats.flatten(),
            )
        )
        if self.obs_timestep_number:
            agent_obs = np.append(agent_obs,
                                  self._episode_steps / self.episode_limit)
        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug("Avail. actions {}".format(
                self.get_avail_agent_actions(agent_id)))
            logging.debug("Move feats {}".format(move_feats))
            logging.debug("Enemy feats {}".format(enemy_feats))
            logging.debug("Ally feats {}".format(ally_feats))
            logging.debug("Own feats {}".format(own_feats))

        return agent_obs

    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent(i) for i in
                      range(self.n_agents + self.n_enemies)]
        return agents_obs

    def get_state(self):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            return obs_concat

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 4 + self.shield_bits_enemy + self.unit_type_bits

        ally_state = np.zeros((self.n_agents, nf_al))
        enemy_state = np.zeros((self.n_enemies, nf_en))

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        for al_id, al_unit in self.agents.items():
            if al_unit.health > 0:
                x = al_unit.pos.x
                y = al_unit.pos.y
                max_cd = self.unit_max_cooldown(al_unit)

                ally_state[al_id, 0] = (
                    al_unit.health / al_unit.health_max
                )  # health
                if (
                    self.map_type == "MMM"
                    and al_unit.unit_type == self.medivac_id
                ):
                    ally_state[al_id, 1] = al_unit.energy / max_cd  # energy
                else:
                    ally_state[al_id, 1] = (
                        al_unit.weapon_cooldown / max_cd
                    )  # cooldown
                # relative X
                ally_state[al_id, 2] = (x - center_x) / self.max_distance_x
                # relative Y
                ally_state[al_id, 3] = (y - center_y) / self.max_distance_y

                ind = 4
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(al_unit)
                    # shield
                    ally_state[al_id, ind] = (al_unit.shield / max_shield)
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(al_unit, True)
                    ally_state[al_id, ind + type_id] = 1

        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                x = e_unit.pos.x
                y = e_unit.pos.y
                max_cd = self.unit_max_cooldown(e_unit)

                # health
                enemy_state[e_id, 0] = (e_unit.health / e_unit.health_max)
                if (self.map_type == "MMM"
                        and e_unit.unit_type == self.medivac_id):
                    # energy
                    enemy_state[e_id, 1] = e_unit.energy / max_cd
                else:
                    # cooldown
                    enemy_state[e_id, 1] = (e_unit.weapon_cooldown / max_cd)
                # relative X
                enemy_state[e_id, 2] = (x - center_x) / self.max_distance_x
                # relative Y
                enemy_state[e_id, 3] = (y - center_y) / self.max_distance_y

                ind = 4
                if self.shield_bits_enemy > 0:
                    max_shield = self.unit_max_shield(e_unit)
                    # shield
                    enemy_state[e_id, ind] = (e_unit.shield / max_shield)
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(e_unit, False)
                    enemy_state[e_id, ind + type_id] = 1

        state = np.append(ally_state.flatten(), enemy_state.flatten())
        state_enemy = np.append(enemy_state.flatten(), ally_state.flatten())
        if self.state_last_action:
            state = np.append(state, self.last_action.flatten())
            last_action_opposite = np.concatenate((self.last_action[self.n_agents:, :],
                                                  self.last_action[:self.n_agents,
                                                  :])).flatten()
            state_enemy = np.append(state_enemy, last_action_opposite.flatten())
        if self.state_timestep_number:
            state = np.append(state,
                              self._episode_steps / self.episode_limit)
            state_enemy = np.append(state_enemy,
                                    self._episode_steps / self.episode_limit)
        state = state.astype(dtype=np.float32)
        state_enemy = state_enemy.astype(dtype=np.float32)

        if self.debug:
            logging.debug("STATE".center(60, "-"))
            logging.debug("Ally state {}".format(ally_state))
            logging.debug("Enemy state {}".format(enemy_state))
            if self.state_last_action:
                logging.debug("Last actions {}".format(self.last_action))
        return state, state_enemy

    def get_obs_size(self):
        """
        Returns the sizes of the observation.
        Due to unit_type_bits, enemy observation can differ from ally ones.
        """
        # Find the dimensions of the matrix containing ally features
        nf_al = 4 + self.unit_type_bits
        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally

        last_action_feats = self.n_actions if self.obs_last_action else 0

        own_feats = self.get_obs_own_feats_size()
        move_feats = self.get_obs_move_feats_size()

        n_enemies, nf_en = self.get_obs_enemy_feats_size()

        enemy_feats = self.n_enemies * nf_en
        ally_feats = self.n_agents * nf_al

        size_for_all = move_feats + enemy_feats + ally_feats + own_feats

        return size_for_all - nf_al + (self.n_agents - 1) * last_action_feats, \
               size_for_all - nf_en + (self.n_enemies - 1) * last_action_feats


    def get_state_size(self):
        """Returns the size of the global state."""
        if self.obs_instead_of_state:
            return self.get_obs_size() * self.n_agents

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 4 + self.shield_bits_enemy + self.unit_type_bits

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = enemy_state + ally_state

        if self.state_last_action:
            size += self.n_agents * self.n_actions
            size += self.n_enemies * self.n_actions
        if self.state_timestep_number:
            size += 1

        return 2, size

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        unit = self.get_unit_by_id(agent_id)
        if unit.health > 0:
            # cannot choose no-op when alive
            avail_actions = [0] * self.n_actions

            # stop should be allowed
            avail_actions[1] = 1

            # see if we can move
            if self.can_move(unit, Direction.NORTH):
                avail_actions[2] = 1
            if self.can_move(unit, Direction.SOUTH):
                avail_actions[3] = 1
            if self.can_move(unit, Direction.EAST):
                avail_actions[4] = 1
            if self.can_move(unit, Direction.WEST):
                avail_actions[5] = 1

            # Can attack only alive units that are alive in the shooting range
            shoot_range = self.unit_shoot_range(agent_id)

            if agent_id < self.n_agents: # allies (set target to enemies)
                target_items = self.enemies.items()
            else: # enemies (set target to allies)
                target_items = self.agents.items()

            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                # Medivacs cannot heal themselves or other flying units
                target_items = [
                    (t_id, t_unit)
                    for (t_id, t_unit) in self.agents.items()
                    if t_unit.unit_type != self.medivac_id
                ]

            for t_id, t_unit in target_items:
                if t_unit.health > 0:
                    dist = self.distance(
                        unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y
                    )
                    if dist <= shoot_range:
                        avail_actions[t_id + self.n_actions_no_attack] = 1

            return avail_actions

        else:
            # only no-op allowed
            return [1] + [0] * (self.n_actions - 1)

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents + self.n_enemies):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def close(self):
        """Close StarCraft II."""
        if self._sc2_proc:
            if type(self._sc2_proc) is list:
                for proc in self._sc2_proc:
                    proc.close()
            else:
                self._sc2_proc.close()

    def _kill_all_units(self):
        """Kill all units on the map."""
        units_alive = [
                          unit.tag for unit in self.agents.values() if
                          unit.health > 0
                      ] + [unit.tag for unit in self.enemies.values() if
                           unit.health > 0]
        debug_command = [
            d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=units_alive))
        ]
        self._controller[0].debug(debug_command)

    def init_units(self):
        """Initialise the units."""
        while True:
            # Sometimes not all units have yet been created by SC2
            self.agents = {}
            self.enemies = {}

            ally_units = []
            for unit in self._obs[0].observation.raw_data.units:
                if unit.owner == 1:
                    ally_units.append(unit)
                    if self._episode_count == 0:
                        self.max_reward_team_2 += unit.health_max + unit.shield_max
            ally_units_sorted = sorted(
                ally_units,
                key=attrgetter("unit_type", "pos.x", "pos.y"),
                reverse=False,
            )

            for i in range(len(ally_units_sorted)):
                self.agents[i] = ally_units_sorted[i]
                if self.debug:
                    logging.debug(
                        "Unit {} is {}, x = {}, y = {}".format(
                            len(self.agents),
                            self.agents[i].unit_type,
                            self.agents[i].pos.x,
                            self.agents[i].pos.y,
                        )
                    )

            enemy_units = []
            for unit in self._obs[1].observation.raw_data.units:
                if unit.owner == 2:
                    enemy_units.append(unit)
                    if self._episode_count == 0:
                        self.max_reward += unit.health_max + unit.shield_max
            enemy_units_sorted = sorted(
                enemy_units,
                key=attrgetter("unit_type", "pos.x", "pos.y"),
                reverse=False,
            )

            for i in range(len(enemy_units_sorted)):
                self.enemies[i] = enemy_units_sorted[i]
                if self.debug:
                    logging.debug(
                        "Enemy unit {} is {}, x = {}, y = {}".format(
                            len(self.enemies),
                            self.enemies[i].unit_type,
                            self.enemies[i].pos.x,
                            self.enemies[i].pos.y,
                        )
                    )

            if self._episode_count == 0:
                all_agents = []
                all_agents.extend(self.agents.values())
                all_agents.extend(self.enemies.values())
                min_unit_type = min(
                    unit.unit_type for unit in all_agents
                )
                self._init_ally_unit_types(min_unit_type)

            all_agents_created = (len(self.agents) == self.n_agents)
            all_enemies_created = (len(self.enemies) == self.n_enemies)

            if all_agents_created and all_enemies_created:  # all good
                return

            try:
                for idx, controller_ in self._controller:
                    controller_.step(1)
                    self._obs[idx] = controller_.observe()
            except (protocol.ProtocolError, protocol.ConnectionError):
                self.full_restart()
                self.reset()

    def update_units(self):
        """
        Update units after an environment step.
        This function assumes that self._obs is up-to-date.
        """
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        self.previous_ally_units = deepcopy(self.agents)
        self.previous_enemy_units = deepcopy(self.enemies)

        for al_id, al_unit in self.agents.items():
            updated = False
            for unit in self._obs[0].observation.raw_data.units:
                if al_unit.tag == unit.tag:
                    self.agents[al_id] = unit
                    updated = True
                    n_ally_alive += 1
                    break

            if not updated:  # dead
                al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs[1].observation.raw_data.units:
                if e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break

            if not updated:  # dead
                e_unit.health = 0

        if (n_ally_alive == 0 and n_enemy_alive > 0
                or self.only_medivac_left(ally=True)):
            return -1  # lost
        if (n_ally_alive > 0 and n_enemy_alive == 0
                or self.only_medivac_left(ally=False)):
            return 1  # won
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0

        return None

    def get_unit_by_id(self, a_id):
        """Get unit by ID."""
        if a_id < self.n_agents:
            # unit is an ally
            return self.agents[a_id]
        else:
            # unit is an enemy
            return self.enemies[a_id - self.n_agents]

    def get_env_info(self):
        env_info = super().get_env_info()
        env_info["n_enemies"] = self.n_enemies
        return env_info

    def get_stats(self):
        stats = {
            "won_team_1": self.games_won_team_1,
            "won_team_2": self.games_won_team_2,
            "battles_draw": self.timeouts,
            "battles_game": self.battles_game,
            "timeouts": self.timeouts,
            "restarts": self.force_restarts,
            "action_error": self.action_error
        }
        return stats
