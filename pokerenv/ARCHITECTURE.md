# PokerEnv Architecture Documentation

## Overview
PokerEnv is a Texas Hold'em poker environment built on OpenAI Gymnasium, designed for training reinforcement learning agents. It supports 2-6 players and provides complete poker game mechanics with hand history logging.

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              POKERENV SYSTEM                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                 TABLE CLASS                                  │
│                            (Inherits gym.Env)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         INITIALIZATION                                │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │ • n_players: 2-6 players                                             │   │
│  │ • player_names: Dictionary mapping player IDs to names               │   │
│  │ • stack_low/stack_high: Random stack bounds (in BBs)                 │   │
│  │ • invalid_action_penalty: Penalty for invalid moves                  │   │
│  │ • hand_history_location: Path for hand history files                 │   │
│  │ • track_single_player: Log only one player's cards (bool)            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         STATE VARIABLES                               │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │ • pot: Current pot size                                              │   │
│  │ • bet_to_match: Amount needed to call                                │   │
│  │ • minimum_raise: Minimum raise amount                                │   │
│  │ • street: PREFLOP → FLOP → TURN → RIVER                             │   │
│  │ • active_players: Count of non-folded players                        │   │
│  │ • current_player_i: Index of current acting player                   │   │
│  │ • next_player_i: Index of next player to act                         │   │
│  │ • last_bet_placed_by: Reference to last bettor                       │   │
│  │ • first_to_act: Reference to first player in round                   │   │
│  │ • street_finished: Boolean flag                                      │   │
│  │ • hand_is_over: Boolean flag                                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         CORE METHODS                                  │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │                                                                        │   │
│  │  reset() → observation                                                │   │
│  │    ├─ Shuffle deck                                                    │   │
│  │    ├─ Randomize player positions                                     │   │
│  │    ├─ Deal hole cards (2 per player)                                 │   │
│  │    ├─ Randomize stack sizes                                          │   │
│  │    ├─ Post blinds (SB: 0.5 BB, BB: 1 BB)                            │   │
│  │    └─ Return observation for first player                            │   │
│  │                                                                        │   │
│  │  step(action) → (observation, rewards, done, info)                   │   │
│  │    ├─ Validate action                                                │   │
│  │    ├─ Execute action (fold/check/call/bet)                           │   │
│  │    ├─ Update pot and player states                                   │   │
│  │    ├─ Check if street/hand finished                                  │   │
│  │    ├─ Transition streets if needed                                   │   │
│  │    ├─ Select next player to act                                      │   │
│  │    └─ Return (obs, rewards, done, info)                              │   │
│  │                                                                        │   │
│  │  _street_transition(transition_to_end=False)                         │   │
│  │    ├─ PREFLOP → FLOP: Deal 3 community cards                        │   │
│  │    ├─ FLOP → TURN: Deal 1 community card                            │   │
│  │    ├─ TURN → RIVER: Deal 1 community card                           │   │
│  │    ├─ RIVER → END: Showdown & pot distribution                      │   │
│  │    └─ Reset betting round variables                                  │   │
│  │                                                                        │   │
│  │  _get_observation(player) → observation vector                       │   │
│  │    └─ Build 58-dimensional observation (see below)                   │   │
│  │                                                                        │   │
│  │  _get_valid_actions(player) → {actions_list, bet_range}             │   │
│  │    └─ Determine legal actions based on game state                    │   │
│  │                                                                        │   │
│  │  _is_action_valid(player, action, valid_actions) → bool             │   │
│  │    └─ Check if action is legal, auto-fold/check if not              │   │
│  │                                                                        │   │
│  │  _distribute_pot()                                                   │   │
│  │    ├─ Calculate hand ranks using Treys evaluator                     │   │
│  │    ├─ Handle side pots for all-in scenarios                         │   │
│  │    └─ Distribute winnings to players                                 │   │
│  │                                                                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                               PLAYER CLASS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         PLAYER ATTRIBUTES                             │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │ • identifier: Unique ID (0-5, persistent across resets)              │   │
│  │ • name: Player name for hand history                                 │   │
│  │ • position: Table position (0=SB, 1=BB, 2-5=others)                  │   │
│  │ • stack: Current chip stack                                          │   │
│  │ • cards: [card1, card2] hole cards                                   │   │
│  │ • state: ACTIVE or FOLDED                                            │   │
│  │ • all_in: Boolean flag                                               │   │
│  │ • bet_this_street: Amount bet in current round                       │   │
│  │ • money_in_pot: Total money contributed                              │   │
│  │ • has_acted: Boolean flag                                            │   │
│  │ • acted_this_street: Boolean flag                                    │   │
│  │ • hand_rank: Hand strength (calculated at showdown)                  │   │
│  │ • winnings: Reward accumulator                                       │   │
│  │ • pending_penalty: Invalid action penalty accumulator                │   │
│  │ • history: List of actions taken in hand                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         PLAYER METHODS                                │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │ • fold(): Mark as folded, record action                              │   │
│  │ • check(): Record check action                                       │   │
│  │ • call(amount): Match bet, handle all-in, return amount              │   │
│  │ • bet(amount): Place bet/raise, handle all-in, return amount         │   │
│  │ • get_reward(): Return accumulated winnings + penalties              │   │
│  │ • punish_invalid_action(): Add penalty to pending_penalty            │   │
│  │ • finish_street(): Reset street-specific flags                       │   │
│  │ • calculate_hand_rank(evaluator, community_cards): Eval hand         │   │
│  │ • reset(): Clear all state for new hand                              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            COMMON ENUMS & CLASSES                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  GameState (IntEnum):        PlayerAction (IntEnum):                         │
│    • PREFLOP = 0              • CHECK = 0                                    │
│    • FLOP = 1                 • FOLD = 1                                     │
│    • TURN = 2                 • BET = 2                                      │
│    • RIVER = 3                • CALL = 3                                     │
│                                                                               │
│  PlayerState (Enum):         TablePosition (IntEnum):                        │
│    • FOLDED = 0               • SB = 0                                       │
│    • ACTIVE = 1               • BB = 1                                       │
│                                                                               │
│  Action Class:                                                               │
│    • action_type: PlayerAction enum                                          │
│    • bet_amount: float (used when action_type is BET)                        │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         OBSERVATION SPACE (58 dims)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Index  │ Content                        │ Notes                             │
│  ───────┼────────────────────────────────┼─────────────────────────────────  │
│  0      │ Acting player identifier       │ Persistent ID (0-5)               │
│  1-4    │ Valid actions mask             │ [CHECK, FOLD, BET, CALL]          │
│  5      │ Valid bet low                  │ Minimum legal bet                 │
│  6      │ Valid bet high                 │ Maximum legal bet                 │
│  7      │ Acting player position         │ 0=SB, 1=BB, etc.                  │
│  8-9    │ Hole card 1                    │ [suit, rank]                      │
│  10-11  │ Hole card 2                    │ [suit, rank]                      │
│  12     │ Acting player stack            │ Chips remaining                   │
│  13     │ Acting player money in pot     │ Total contributed                 │
│  14     │ Acting player bet this street  │ Current round bet                 │
│  15     │ Current street                 │ 0=PREFLOP, 1=FLOP, 2=TURN, 3=RIVER│
│  16-17  │ Community card 1               │ [suit, rank] (0 if not dealt)     │
│  18-19  │ Community card 2               │ [suit, rank] (0 if not dealt)     │
│  20     │ Pot size                       │ Total chips in pot                │
│  21     │ Bet to match                   │ Amount to call                    │
│  22     │ Minimum raise                  │ Min raise above bet_to_match      │
│  23-28  │ Opponent 1 info                │ [pos, state, stack, pot, bet, all]│
│  29-34  │ Opponent 2 info                │ [pos, state, stack, pot, bet, all]│
│  35-40  │ Opponent 3 info                │ [pos, state, stack, pot, bet, all]│
│  41-46  │ Opponent 4 info                │ [pos, state, stack, pot, bet, all]│
│  47-52  │ Opponent 5 info                │ [pos, state, stack, pot, bet, all]│
│  53-58  │ (Padding for max 6 players)    │                                   │
│                                                                               │
│  Opponent Info (6 values each):                                              │
│    [0] Position, [1] State (0=FOLDED, 1=ACTIVE), [2] Stack,                 │
│    [3] Money in pot, [4] Bet this street, [5] All-in flag                   │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            ACTION SPACE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Tuple(Discrete(4), Box(-inf, inf, (1,1)))                                  │
│                                                                               │
│  Component 1: Action Type (Discrete)                                         │
│    • 0 = CHECK                                                               │
│    • 1 = FOLD                                                                │
│    • 2 = BET/RAISE                                                           │
│    • 3 = CALL                                                                │
│                                                                               │
│  Component 2: Bet Amount (Continuous)                                        │
│    • Used only when action type = 2 (BET)                                   │
│    • Must be within [valid_bet_low, valid_bet_high]                         │
│    • Represents total bet this street (not raise amount)                    │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              REWARD SYSTEM                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Reward Structure:                                                           │
│    • Rewards returned as numpy array: [r0, r1, r2, r3, r4, r5]             │
│    • Index corresponds to player.identifier (NOT position)                  │
│    • Rewards only non-None when player has acted or hand ends               │
│                                                                               │
│  Reward Components:                                                          │
│    reward = winnings - money_lost + pending_penalty                          │
│                                                                               │
│    • winnings: Money won from pot (distributed at hand end)                 │
│    • money_lost: Money put into pot during hand                             │
│    • pending_penalty: Negative reward for invalid actions                   │
│                                                                               │
│  Key Points:                                                                 │
│    • Sparse rewards: Most steps return 0, final pot distribution at end     │
│    • Invalid action penalty applied immediately if enabled                  │
│    • Side pots handled automatically for all-in scenarios                   │
│    • Player identifier persists across table.reset() calls                  │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            GAME FLOW DIAGRAM                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   table.reset()                                                              │
│        │                                                                      │
│        ├──> Shuffle deck                                                     │
│        ├──> Randomize player order                                           │
│        ├──> Deal hole cards                                                  │
│        ├──> Randomize stacks                                                 │
│        ├──> Post blinds (SB, BB)                                             │
│        └──> Return observation for first player after BB                     │
│                                                                               │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                     BETTING ROUND LOOP                           │       │
│   │                                                                  │       │
│   │  table.step(action)                                              │       │
│   │       │                                                          │       │
│   │       ├──> Validate action                                       │       │
│   │       │     └──> If invalid: auto-fold or auto-check + penalty   │       │
│   │       │                                                          │       │
│   │       ├──> Execute valid action                                  │       │
│   │       │     ├──> FOLD: Remove from active_players               │       │
│   │       │     ├──> CHECK: Mark as acted                           │       │
│   │       │     ├──> CALL: Match bet_to_match                        │       │
│   │       │     └──> BET: Update pot, bet_to_match, minimum_raise    │       │
│   │       │                                                          │       │
│   │       ├──> Check termination conditions                          │       │
│   │       │     ├──> Only 1 player not folded? → Hand over          │       │
│   │       │     ├──> All active all-in? → Run out remaining streets  │       │
│   │       │     └──> Betting round complete? → Next street           │       │
│   │       │                                                          │       │
│   │       ├──> Select next player                                    │       │
│   │       │     └──> Skip folded and all-in players                  │       │
│   │       │                                                          │       │
│   │       └──> Return (observation, rewards, done, info)             │       │
│   │                                                                  │       │
│   └──────────────────────────────────────────────────────────────────┘       │
│        │                                                                      │
│        ├──> If street_finished → _street_transition()                        │
│        │                                                                      │
│        └──> If hand_is_over → _distribute_pot() + _finish_hand()             │
│                                                                               │
│   Street Transitions:                                                        │
│        PREFLOP → FLOP (deal 3 cards)                                         │
│             │                                                                 │
│        FLOP → TURN (deal 1 card)                                             │
│             │                                                                 │
│        TURN → RIVER (deal 1 card)                                            │
│             │                                                                 │
│        RIVER → SHOWDOWN (evaluate hands, distribute pot)                     │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         HAND HISTORY SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Purpose: Generate PokerStars-compatible hand history files                  │
│                                                                               │
│  Configuration:                                                              │
│    • hand_history_enabled: Toggle on/off (default: False)                   │
│    • hand_history_location: Directory path for output files                 │
│    • track_single_player: If True, only log player 0's cards                │
│                                                                               │
│  File Format:                                                                │
│    ┌────────────────────────────────────────────────────────────┐           │
│    │ PokerStars Hand #12345: Hold'em No Limit ($2.50/$5.00)    │           │
│    │ Table 'Wempe III' 6-max Seat #2 is the button             │           │
│    │ Seat 1: Player1 ($500.00 in chips)                        │           │
│    │ Seat 2: Player2 ($750.00 in chips)                        │           │
│    │ ...                                                        │           │
│    │ *** HOLE CARDS ***                                        │           │
│    │ Dealt to TrackedAgent1 [Ah Kd]                           │           │
│    │ Player1: folds                                            │           │
│    │ Player2: raises $15.00 to $20.00                          │           │
│    │ *** FLOP *** [7h 8c 9s]                                   │           │
│    │ ...                                                        │           │
│    │ *** TURN *** [7h 8c 9s] [Td]                              │           │
│    │ *** RIVER *** [7h 8c 9s Td] [Jc]                          │           │
│    │ *** SHOW DOWN ***                                         │           │
│    │ Player2: shows [Ah Kd] (a straight, Seven to Jack)       │           │
│    │ Player2 collected $125.00 from pot                        │           │
│    │ *** SUMMARY ***                                           │           │
│    │ Total pot $125.00 | Rake $0.00                            │           │
│    │ Board [7h 8c 9s Td Jc]                                    │           │
│    └────────────────────────────────────────────────────────────┘           │
│                                                                               │
│  Usage:                                                                      │
│    • Compatible with PokerTracker, Hold'em Manager                          │
│    • Useful for analyzing agent behavior                                    │
│    • Enable periodically to monitor training progress                       │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      INVALID ACTION HANDLING                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Philosophy: Never raise exceptions for invalid actions during training      │
│                                                                               │
│  When Invalid Action Occurs:                                                 │
│                                                                               │
│    1. Check if action type is in valid_actions list                         │
│       └─> If not: Auto-fold (if possible) or Auto-check                     │
│                                                                               │
│    2. If action type is BET, check bet amount                               │
│       └─> If outside [valid_bet_low, valid_bet_high]:                       │
│             Auto-fold (if possible) or Auto-check                            │
│                                                                               │
│    3. Apply invalid_action_penalty (if configured)                          │
│       └─> Added to player.pending_penalty                                   │
│       └─> Subtracted from reward in next get_reward() call                  │
│                                                                               │
│  Invalid Action Masking:                                                     │
│    • Observation contains valid_actions mask (indices 1-4)                  │
│    • Observation contains valid_bet_range (indices 5-6)                     │
│    • Agents can use these to avoid invalid actions                          │
│                                                                               │
│  Benefits:                                                                   │
│    • Training never crashes from invalid actions                            │
│    • Agent learns to avoid invalid moves (via penalty)                      │
│    • Environment handles edge cases gracefully                              │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         IMPORTANT DESIGN NOTES                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  1. Player Identifiers vs. Positions                                         │
│     • identifier: Permanent ID (0-5), persistent across resets              │
│     • position: Table position (0=SB, 1=BB, etc.), changes each hand        │
│     • Observation contains identifier, allowing agent to track itself       │
│                                                                               │
│  2. Player Rotation                                                          │
│     • Players shuffled at start of each hand (reset())                      │
│     • Each agent experiences all positions over time                        │
│     • Ensures agents don't overfit to specific positions                    │
│                                                                               │
│  3. Stack Size Randomization                                                 │
│     • Stacks randomized in [stack_low, stack_high] BBs each reset           │
│     • Exposes agents to various stack depth scenarios                       │
│     • Important for robust strategy learning                                │
│                                                                               │
│  4. Reward Structure                                                         │
│     • Sparse: Most steps return 0, rewards at hand end                      │
│     • Indexed by player.identifier, not position                            │
│     • agents[identifier].rewards.append(reward[identifier])                 │
│                                                                               │
│  5. Betting Round Completion                                                 │
│     • Round ends when action returns to last_bet_placed_by                  │
│     • Or when only 1 active non-all-in player remains                       │
│     • Handles heads-up (2-player) correctly                                 │
│                                                                               │
│  6. All-In Handling                                                          │
│     • All-in players skip turn selection                                    │
│     • If all active players all-in, streets run out with no actions         │
│     • Side pots calculated correctly in _distribute_pot()                   │
│                                                                               │
│  7. Dependencies                                                             │
│     • treys: Card evaluation and hand ranking                               │
│     • gymnasium: Environment interface                                      │
│     • numpy: Array operations and RNG                                       │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         TYPICAL USAGE PATTERN                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  1. Initialize environment and agents                                        │
│     ```python                                                                │
│     table = Table(n_players=6, invalid_action_penalty=-1)                   │
│     agents = [YourAgent() for _ in range(6)]                                │
│     ```                                                                      │
│                                                                               │
│  2. Training loop                                                            │
│     ```python                                                                │
│     while training:                                                          │
│         obs = table.reset()  # New hand                                     │
│         acting_player_id = int(obs[0])                                       │
│                                                                               │
│         while True:  # Betting loop                                          │
│             action = agents[acting_player_id].get_action(obs)               │
│             obs, rewards, done, _ = table.step(action)                      │
│                                                                               │
│             if done:                                                         │
│                 # Distribute final rewards                                   │
│                 for i in range(6):                                           │
│                     agents[i].store_reward(rewards[i])                      │
│                 break                                                        │
│             else:                                                            │
│                 # Intermediate reward (usually 0 or penalty)                │
│                 agents[acting_player_id].store_reward(                      │
│                     rewards[acting_player_id]                               │
│                 )                                                            │
│                 acting_player_id = int(obs[0])                              │
│     ```                                                                      │
│                                                                               │
│  3. Periodic hand history logging                                            │
│     ```python                                                                │
│     if iteration % 1000 == 0:                                               │
│         table.hand_history_enabled = True                                   │
│         # ... play hands ...                                                 │
│         table.hand_history_enabled = False                                  │
│     ```                                                                      │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         KEY OBSERVATION INDICES                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  From obs_indices.py:                                                        │
│                                                                               │
│    ACTING_PLAYER = 0           # Current player's identifier                │
│    VALID_ACTIONS = [1, 2, 3, 4]  # Binary mask for valid actions           │
│    VALID_BET_LOW = 5           # Minimum bet/raise allowed                  │
│    VALID_BET_HIGH = 6          # Maximum bet/raise allowed                  │
│    ACTING_PLAYER_POSITION = 7  # Position at table (0=SB, 1=BB, etc.)      │
│    ACTING_PLAYER_STACK_SIZE = 12  # Remaining chips                         │
│    POT_SIZE = 20               # Current pot size                           │
│                                                                               │
│  Usage in agent:                                                             │
│    ```python                                                                 │
│    import pokerenv.obs_indices as indices                                   │
│                                                                               │
│    my_id = obs[indices.ACTING_PLAYER]                                       │
│    valid_actions = obs[indices.VALID_ACTIONS]                               │
│    min_bet = obs[indices.VALID_BET_LOW]                                     │
│    max_bet = obs[indices.VALID_BET_HIGH]                                    │
│    my_stack = obs[indices.ACTING_PLAYER_STACK_SIZE]                         │
│    pot = obs[indices.POT_SIZE]                                              │
│    ```                                                                       │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘