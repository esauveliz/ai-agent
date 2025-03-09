import os
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from mistralai import Mistral
import discord
import aiohttp
import logging
from bs4 import BeautifulSoup
import time
from datetime import datetime

# Setup logging
logger = logging.getLogger("discord")

from enum import Enum

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = """You are a fantasy basketball expert assistant. You have deep knowledge of NBA players, their statistics, 
fantasy basketball strategy, and current NBA trends. You provide analytical and data-driven advice while considering factors 
like player performance, injuries, team dynamics, and schedule."""

COMPARE_PROMPT = """You are a fantasy basketball expert. Analyze and compare the provided NBA players for fantasy basketball purposes. 
Pay special attention to CURRENT injuries, recent performance, and team situations.

Current information about the players:
{current_info}

Players to compare: {players}

Format your response as follows:
1. First mention if any names are invalid/inactive or if there aren't enough names
2. If all names are valid, provide the ranking with explanations, heavily weighing current injuries and availability
3. Keep explanations concise but informative, focusing on CURRENT situation and value
4. If a player is currently injured, this MUST be mentioned first in their analysis"""

MAX_HISTORY = 10  # Maximum number of messages to keep in history per channel

class Position(Enum):
    PG = "Point Guard"
    SG = "Shooting Guard"
    SF = "Small Forward"
    PF = "Power Forward"
    C = "Center"
    G = "Guard"
    F = "Forward"
    UTIL = "Utility"

class DraftState:
    def __init__(self, total_rounds: int, pick_position: int, total_players: int):
        self.total_rounds = total_rounds
        self.pick_position = pick_position
        self.total_players = total_players  # Total players in draft
        self.current_round = 1
        self.picks_made = 0  # Track total picks made
        self.drafted_players: List[Tuple[str, Position]] = []  # List of (player, position) tuples
        self.my_team: List[Tuple[str, Position]] = []  # Track my drafted players and their positions
        self.available_players = []
        self.is_active = False
        
    def is_user_turn(self) -> bool:
        """Check if it's the user's turn to draft"""
        # Calculate the current pick in the round (1-based)
        current_pick = (self.picks_made % self.total_players) + 1
        return current_pick == self.pick_position
    
    def get_roster_needs(self) -> Dict[Position, int]:
        """Calculate roster needs based on typical fantasy basketball roster requirements"""
        roster_requirements = {
            Position.PG: 2,
            Position.SG: 2,
            Position.SF: 2,
            Position.PF: 2,
            Position.C: 2,
            Position.UTIL: 3
        }
        
        # Count current roster by position
        current_roster = defaultdict(int)
        for _, pos in self.my_team:
            current_roster[pos] += 1
            
        # Calculate remaining needs
        needs = {}
        for pos, req in roster_requirements.items():
            needs[pos] = max(0, req - current_roster[pos])
            
        return needs
    
    def update_draft(self, drafted_player: str, position: Position) -> None:
        """Update draft state after a pick"""
        self.picks_made += 1
        if drafted_player in self.available_players:
            self.available_players.remove(drafted_player)
            
        # Record the pick
        self.drafted_players.append((drafted_player, position))
        
        # If it was our pick, add to my_team
        if self.is_user_turn():
            self.my_team.append((drafted_player, position))
            
        # Update round if necessary
        self.current_round = (self.picks_made // self.total_players) + 1
            
        # Check if draft is complete
        if self.picks_made >= (self.total_players * self.total_rounds):
            self.is_active = False

class MistralAgent:
    def __init__(self):
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.channel_history: Dict[int, List[dict]] = defaultdict(list)
        self.draft_states: Dict[int, DraftState] = {}
        self.cached_players: Dict[str, Dict[str, str]] = {}  # Map of player name to their attributes
        self.last_fetch_time = 0  # Track when we last fetched players
        self.ESPN_PLAYER_IDS = {
            # Stars and popular players
            "Nikola Jokic": "3112335",
            "LeBron James": "1966",
            "Luka Doncic": "3945274",
            "Stephen Curry": "3975",
            "Kevin Durant": "3202",
            "Giannis Antetokounmpo": "3032977",
            "Joel Embiid": "3059318",
            "Jayson Tatum": "4065648",
            "Devin Booker": "3136193",
            "Shai Gilgeous-Alexander": "4278073",
            "Anthony Davis": "6583",
            "Damian Lillard": "6606",
            "Donovan Mitchell": "3908809",
            "Ja Morant": "4279888",
            "Trae Young": "4277905",
            "Anthony Edwards": "4594268",
            "Victor Wembanyama": "5088141",
            "Tyrese Haliburton": "4395651",
            "Kawhi Leonard": "6450",
            "Paul George": "4251",
            # Add more as needed
        }

    def _update_history(self, channel_id: int, role: str, content: str):
        """Add a message to the channel's history and maintain history size."""
        self.channel_history[channel_id].append({"role": role, "content": content})
        # Keep only the last MAX_HISTORY messages
        if len(self.channel_history[channel_id]) > MAX_HISTORY:
            self.channel_history[channel_id] = self.channel_history[channel_id][-MAX_HISTORY:]

    async def run(self, message: discord.Message):
        # Get the channel's conversation history
        channel_id = message.channel.id
        
        # Construct messages list with system prompt and history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.channel_history[channel_id])
        messages.append({"role": "user", "content": message.content})

        # Get response from Mistral
        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )
        
        # Update history with both user message and assistant's response
        self._update_history(channel_id, "user", message.content)
        response_content = response.choices[0].message.content
        self._update_history(channel_id, "assistant", response_content)

        return response_content

    def _find_player_match(self, search_name: str, players_map: Dict[str, Dict[str, str]]) -> Optional[Tuple[str, Dict[str, str]]]:
        """Find the best match for a player name in the players map."""
        def normalize_name(name: str) -> str:
            """Normalize a name by removing special characters and converting to lowercase"""
            # Remove periods, apostrophes, and other special characters
            name = ''.join(c for c in name if c.isalnum() or c.isspace())
            return name.lower().strip()
        
        def split_name(name: str) -> str:
            """Split a name that might be camelCase or PascalCase"""
            # Common prefixes to handle
            prefixes = ["Mc", "Mac", "De", "Van", "Von", "St", "O", "Le", "La", "Al"]
            
            # First try to split on capital letters
            parts = []
            current = []
            for i, char in enumerate(name):
                if i > 0 and char.isupper():
                    # Check if it's part of a prefix
                    current_word = "".join(current)
                    next_chars = name[i:i+3] if i+3 <= len(name) else name[i:]
                    is_prefix = False
                    for prefix in prefixes:
                        if (current_word + next_chars).startswith(prefix):
                            is_prefix = True
                            break
                    
                    if not is_prefix:
                        parts.append("".join(current))
                        current = []
                current.append(char)
            parts.append("".join(current))
            
            return " ".join(parts)
        
        # Clean and prepare search name
        search_name = search_name.strip()
        if " " not in search_name:
            search_name = split_name(search_name)
        normalized_search = normalize_name(search_name)
        search_parts = normalized_search.split()
        
        # Try exact match first
        for player_name, stats in players_map.items():
            if normalize_name(player_name) == normalized_search:
                return (player_name, stats)
        
        # Try matching first and last name
        if len(search_parts) > 1:
            for player_name, stats in players_map.items():
                normalized_player = normalize_name(player_name)
                player_parts = normalized_player.split()
                if len(player_parts) > 1:
                    # Match both first and last name
                    if (search_parts[0] in player_parts[0] and 
                        search_parts[-1] in player_parts[-1]):
                        return (player_name, stats)
        
        # Try matching without spaces
        no_space_search = "".join(search_parts)
        for player_name, stats in players_map.items():
            no_space_player = "".join(normalize_name(player_name).split())
            if no_space_search in no_space_player:
                return (player_name, stats)
        
        # Try partial matches with score-based ranking
        best_match = None
        best_score = 0
        for player_name, stats in players_map.items():
            normalized_player = normalize_name(player_name)
            player_parts = normalized_player.split()
            
            # Calculate match score
            score = 0
            for search_part in search_parts:
                # Full word match
                if search_part in player_parts:
                    score += 3
                # Partial word match
                elif any(search_part in part for part in player_parts):
                    score += 1
                    
            if score > best_score:
                best_score = score
                best_match = (player_name, stats)
        
        # Only return partial matches if they have a minimum score
        if best_score >= 2:
            return best_match
            
        return None

    async def compare_players(self, players: List[str]) -> str:
        """Compare NBA players for fantasy basketball purposes."""
        if len(players) < 2:
            return "Please provide at least 2 players to compare."
            
        players_str = ", ".join(players)
        
        # First get current rankings from HashtagBasketball
        current_time = int(time.time())
        if not self.cached_players or (current_time - self.last_fetch_time) > 3600:
            self.cached_players = await self.fetch_players_list()
            self.last_fetch_time = current_time
            
        # Get rankings for requested players with better name matching
        player_rankings = {}
        unmatched_players = []
        for player in players:
            match = self._find_player_match(player, self.cached_players)
            if match:
                player_name, stats = match
                # Get numerical rank from stats
                try:
                    rank = int(stats.get('RANK', '999'))
                except ValueError:
                    rank = 999
                player_rankings[player] = {
                    "exact_name": player_name,
                    "stats": stats,
                    "rank": rank,
                    "injury_status": "Unknown"  # Will be updated with real-time check
                }
            else:
                unmatched_players.append(player)
        
        # If any players weren't found in rankings, note this
        if unmatched_players:
            unmatched_str = ", ".join(unmatched_players)
            logger.warning(f"Could not find rankings for players: {unmatched_str}")
                
        # Get current injury/news info for each player and update rankings
        search_results = []
        for player in players:
            # First search specifically for season-ending injuries
            season_ending_search = f"{player} NBA season ending injury 2024-25 out for season"
            season_ending_result = await self.web_search(season_ending_search)
            
            # Then search for current injury status and news
            injury_search = f"{player} NBA injury status March 2025 current"
            injury_result = await self.web_search(injury_search)
            
            # Finally get recent performance
            performance_search = f"{player} NBA fantasy basketball performance March 2025"
            performance_result = await self.web_search(performance_search)
            
            # Combine with HashtagBasketball data if available
            player_info = "Current Status:\n"
            if player in player_rankings:
                exact_name = player_rankings[player]["exact_name"]
                stats = player_rankings[player]["stats"]
                player_info += f"HashtagBasketball Rankings (as {exact_name}):\n"
                for stat, value in stats.items():
                    player_info += f"- {stat}: {value}\n"
                
                # Update player's injury status
                if "season" in season_ending_result.lower() and "ending" in season_ending_result.lower():
                    player_rankings[player]["injury_status"] = "Season-Ending"
                    player_rankings[player]["rank"] = 9999  # Force to bottom
                elif any(term in injury_result.lower() for term in ["out indefinitely", "out for", "expected to miss", "several weeks"]):
                    player_rankings[player]["injury_status"] = "Long-Term"
                    player_rankings[player]["rank"] += 50  # Significantly lower ranking
                elif any(term in injury_result.lower() for term in ["day-to-day", "questionable", "probable"]):
                    player_rankings[player]["injury_status"] = "Day-to-Day"
                    # Keep original ranking mostly intact
                else:
                    player_rankings[player]["injury_status"] = "Healthy"
            else:
                player_info += "Not found in current HashtagBasketball rankings\n"
            
            player_info += f"\nSeason-Ending Injury Check:\n{season_ending_result}\n"
            player_info += f"\nCurrent Injury Status:\n{injury_result}\n"
            player_info += f"\nRecent Performance:\n{performance_result}"
            
            search_results.append(f"Information for {player}:\n{player_info}\n")
        
        # Sort players by adjusted rank (considering injuries)
        sorted_players = sorted(
            [(p, info) for p, info in player_rankings.items()],
            key=lambda x: x[1]["rank"]
        )
        
        # Create ranking summary
        ranking_summary = "\nFinal Rankings (considering injuries):\n"
        for i, (player, info) in enumerate(sorted_players, 1):
            status = f" ({info['injury_status']})" if info['injury_status'] != "Healthy" else ""
            ranking_summary += f"{i}. {info['exact_name']}{status}\n"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nIMPORTANT RULES:\n1. Season-ending injuries MUST be mentioned first and ranked last\n2. Long-term injuries (1+ month) should be ranked lower but not last\n3. Day-to-day injuries should be mentioned but not significantly affect ranking\n4. For healthy players, use HashtagBasketball rankings\n\nPlayer Information:\n" + "\n".join(search_results) + ranking_summary},
            {"role": "user", "content": f"Compare these players for fantasy basketball value RIGHT NOW. Focus heavily on injury status - especially season-ending injuries like Joel Embiid's knee surgery. Explain why each player is ranked where they are, mentioning both their baseline value and any injury concerns: {players_str}"}
        ]

        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )

        response_content = response.choices[0].message.content
        if len(response_content) > 1900:
            response_content = response_content[:1900] + "..."

        return response_content

    async def _get_players(self, force_refresh: bool = False) -> Dict[str, Dict[str, str]]:
        """Get players from cache if available and fresh, otherwise fetch new data.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            Dict mapping player names to their attributes
        """
        current_time = int(time.time())
        
        # Use cache if available and less than 1 hour old, unless force refresh is requested
        if not force_refresh and self.cached_players and (current_time - self.last_fetch_time) <= 3600:
            logger.info("Using cached player rankings")
            return self.cached_players
            
        # Fetch fresh data
        logger.info("Fetching fresh player rankings")
        players = await self.fetch_players_list()
        
        # Update cache if fetch was successful
        if players:
            self.cached_players = players
            self.last_fetch_time = current_time
            
        return players

    async def fetch_players_list(self) -> Dict[str, Dict[str, str]]:
        """Fetch current NBA players list from hashtagbasketball.com
        
        Returns:
            Dict mapping player names to their attributes
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        try:
            players_map = {}
            url = "https://hashtagbasketball.com/fantasy-basketball-rankings"
            
            logger.info("Starting to fetch top 215 players from hashtagbasketball.com")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch players: {response.status}")
                        return {}
                    
                    logger.info("Successfully got response from server")
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find the rankings table
                    rankings_table = soup.find('table', {'id': 'ContentPlaceHolder1_GridView1'})
                    
                    if not rankings_table:
                        logger.warning("Could not find rankings table")
                        return {}
                    
                    # Get headers
                    header_row = rankings_table.find('tr')
                    if not header_row:
                        logger.warning("Could not find table headers")
                        return {}
                    
                    headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
                    
                    # Extract player data from table rows
                    rows = rankings_table.find_all('tr')[1:]  # Skip header row
                    if not rows:
                        logger.warning("No player rows found in rankings table")
                        return {}
                    
                    # Extract player data from table rows
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:  # Make sure we have enough cells
                            player_name = cells[1].get_text(strip=True)  # Player name is in second column
                            if player_name and player_name != "PLAYER":  # Skip entries that are just "PLAYER"
                                # Create dictionary of all stats for this player
                                stats = {}
                                for i, cell in enumerate(cells):
                                    if i < len(headers):  # Make sure we have a header for this column
                                        stats[headers[i]] = cell.get_text(strip=True)
                                players_map[player_name] = stats
                    
                    if not players_map:
                        logger.warning("No players found in rankings")
                        return {}
                    
                    initial_count = len(players_map)
                    # Filter out any remaining "PLAYER" entries
                    players_map = {name: stats for name, stats in players_map.items() if name != "PLAYER"}
                    if len(players_map) != initial_count:
                        logger.info(f"Removed {initial_count - len(players_map)} invalid player entries")
                    
                    logger.info(f"Successfully fetched {len(players_map)} players from hashtagbasketball.com")
                    return players_map
                    
        except Exception as e:
            logger.error(f"Error fetching players from hashtagbasketball.com: {str(e)}")
            return {}

    async def start_draft(self, channel_id: int, total_rounds: int, pick_position: int, total_players: int) -> str:
        """Initialize a new draft session"""
        # Check if there's already an active draft
        if channel_id in self.draft_states and self.draft_states[channel_id].is_active:
            return "A draft is already in progress in this channel! Use !pick to record picks."
        
        draft_state = DraftState(total_rounds, pick_position, total_players)
        
        try:
            # Fetch initial player list
            players_with_stats = await self._get_players()
            draft_state.available_players = [player[0] for player in players_with_stats.items()]
            
            if not draft_state.available_players:
                return "Error: Could not fetch player list. Please try again later."
            
            # Activate the draft and store the state
            draft_state.is_active = True
            self.draft_states[channel_id] = draft_state
            
            # Return success message with draft info
            return f"""🏀 Draft Successfully Started! 🏀

Draft Settings:
• Position: Pick {pick_position} of {total_players}
• Length: {total_rounds} rounds
• Total Picks: {total_rounds * total_players}

Available Commands:
• !pick <pick_number> <player_name> <position> - When it's your turn, you MUST specify a position to assign the player to your lineup
• !pick <pick_number> <player_name> [position] - For other picks, position is optional
• !getrec - Get draft recommendations based on current draft state
• !myteam - View your current roster
• !players - View available players

Valid positions for your picks:
• PG - Point Guard
• SG - Shooting Guard
• SF - Small Forward
• PF - Power Forward
• C - Center
• UTIL - Utility

Your pick will come up every {total_players} picks. Good luck! 🎯"""
            
        except Exception as e:
            logger.error(f"Error starting draft: {str(e)}")
            return f"Error starting draft: {str(e)}\nPlease try again or contact support if the issue persists."

    async def update_draft_pick(self, channel_id: int, pick_num: int, player_name: str, position: Optional[str] = None) -> str:
        """Update draft state with a new pick"""
        if channel_id not in self.draft_states or not self.draft_states[channel_id].is_active:
            return "No active draft in this channel!"
        
        draft_state = self.draft_states[channel_id]
        
        try:
            if pick_num < 1:
                return "Pick number must be positive!"
            
            if not player_name:
                return "Please provide a player name!"
            
            # Calculate if this pick corresponds to the user's position
            is_user_pick = pick_num == draft_state.pick_position
            
            # If it's user's pick, require position
            if is_user_pick and not position:
                return "Please specify a position for your pick (PG/SG/SF/PF/C/UTIL)."
            
            # Convert position string to enum if provided
            pos_enum = None
            if position:
                try:
                    pos_enum = Position[position]
                except KeyError:
                    return f"Invalid position '{position}'. Please use: PG, SG, SF, PF, C, or UTIL"
            
            # Find the best matching player in available players
            best_match = None
            for player in draft_state.available_players:
                if player_name.lower() in player.lower():
                    if best_match is None or len(player) < len(best_match):
                        best_match = player
                # Also check first/last name exact matches
                player_parts = player.lower().split()
                if player_name.lower() in player_parts:
                    best_match = player
                    break
            
            # If no match found, use the provided name as is
            if not best_match:
                logger.info(f"Player '{player_name}' not found in available players list, using name as provided")
                best_match = player_name.title()  # Convert to title case for consistency
            else:
                # Remove the player from available players only if they were in our list
                draft_state.available_players.remove(best_match)
            
            # Record the pick and update round
            draft_state.picks_made += 1
            # Calculate current round (1-based)
            draft_state.current_round = ((draft_state.picks_made - 1) // draft_state.total_players) + 1
            
            # Add to drafted players and user's team if it's their pick
            draft_state.drafted_players.append((best_match, pos_enum))
            if is_user_pick:
                draft_state.my_team.append((best_match, pos_enum))
            
            # Check if draft is complete
            if draft_state.picks_made >= (draft_state.total_players * draft_state.total_rounds):
                draft_state.is_active = False
                return "🎉 Draft Complete! 🎉"
            
            # If it's the next user's turn
            if draft_state.is_user_turn():
                return f"""✅ Pick #{pick_num}: {best_match}{f" ({position})" if position else ""}

Draft Status:
• Round: {draft_state.current_round}/{draft_state.total_rounds}
• Pick: {pick_num}/{draft_state.total_players}

🎯 It's your turn to draft! Use !getrec for recommendations.

To make your pick, use:
!pick <pick_number> <player_name> <position>

Available positions:
• PG - Point Guard
• SG - Shooting Guard
• SF - Small Forward
• PF - Power Forward
• C - Center
• UTIL - Utility"""
            
            return f"""✅ Pick #{pick_num}: {best_match}{f" ({position})" if position else ""}

Draft Status:
• Round: {draft_state.current_round}/{draft_state.total_rounds}
• Pick: {pick_num}/{draft_state.total_players}"""
            
        except ValueError:
            return "Invalid pick number! Please provide a valid number."

    def _split_into_messages(self, content: str, chunk_size: int = 1900) -> List[str]:
        """Split a long message into chunks that fit within Discord's message limit.
        
        Args:
            content: The full message content to split
            chunk_size: Maximum size of each chunk (default 1900 to leave room for Discord's limit of 2000)
            
        Returns:
            List of message chunks
        """
        responses = []
        current_chunk = ""
        
        for line in content.split('\n'):
            # If adding this line would exceed Discord's limit, start a new chunk
            if len(current_chunk) + len(line) + 1 > chunk_size:
                responses.append(current_chunk)
                current_chunk = line
            else:
                if current_chunk:
                    current_chunk += '\n'
                current_chunk += line
        
        # Add the last chunk if it's not empty
        if current_chunk:
            responses.append(current_chunk)
            
        return responses

    async def get_draft_recommendation(self, channel_id: int) -> List[str]:
        """Get draft recommendations based on draft position and current state"""
        if channel_id not in self.draft_states or not self.draft_states[channel_id].is_active:
            return ["No active draft in this channel!"]
        
        draft_state = self.draft_states[channel_id]
        
        # Calculate picks until next turn
        current_pick = (draft_state.picks_made % draft_state.total_players) + 1
        picks_until_turn = 0
        if current_pick <= draft_state.pick_position:
            picks_until_turn = draft_state.pick_position - current_pick
        else:
            picks_until_turn = (draft_state.total_players - current_pick) + draft_state.pick_position
        
        # Get roster needs
        needs = draft_state.get_roster_needs()
        needs_str = "Current roster needs:\n"
        for pos, count in needs.items():
            if count > 0:
                needs_str += f"• {pos.value}: {count} needed\n"
        
        # Get current information about available players
        available_players_str = ", ".join(draft_state.available_players[:10])
        
        # Use web search to get current information about top available players
        search_term = f"{available_players_str} NBA fantasy basketball rankings current stats injuries 2024"
        web_search_result = await self.web_search(search_term)
        
        # Draft position context
        picks_context = ""
        if draft_state.is_user_turn():
            picks_context = "🎯 It's currently your turn to draft!"
        else:
            picks_context = f"You pick in {picks_until_turn} picks"
        
        messages = [
            {"role": "system", "content": f"""You are a fantasy basketball draft expert. Consider:
1. Current round: {draft_state.current_round}/{draft_state.total_rounds}
2. Draft position: Pick {draft_state.pick_position} of {draft_state.total_players}
3. Current pick in round: {current_pick}/{draft_state.total_players}
4. Picks until your turn: {picks_until_turn}
5. My team so far: {', '.join([f"{player}" for player, _ in draft_state.my_team])}
6. Roster needs:
{needs_str}

Current player information:
{web_search_result}

Provide draft recommendations in strictly in the following format:
1. Draft Position Analysis: 2-3 sentences about the current draft state and picks until your turn
2. Draft Strategy: 2-3 sentences about what to prioritize based on draft position, current pick, and team needs
3. Top Available Players: List 5 best available players with 1-line explanation for each
4. Recommendations:
   - If it's your turn: State best pick and second-best pick with brief reasoning
   - If not your turn: List 2-3 players you hope will still be available at your pick

Focus on:
- Draft position strategy (who might be available at your pick)
- Team needs and roster construction
- Position scarcity
- Injury status and availability
- Recent performance and trends"""},
            {"role": "user", "content": f"Analyze the draft situation and recommend players considering {picks_context}. Available players include: {available_players_str}"}
        ]

        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )

        full_response = f"""Draft Analysis:
{picks_context}

{needs_str}
{response.choices[0].message.content}"""
        return self._split_into_messages(full_response)

    async def web_search(self, query: str) -> str:
        """Perform a web search using the web_search tool."""
        try:
            # Use the web_search tool
            search_result = await self.client.chat.complete_async(
                model=MISTRAL_MODEL,
                messages=[{
                    "role": "system", 
                    "content": "You are searching for current NBA player information. Return only factual, relevant information."
                }, {
                    "role": "user",
                    "content": query
                }]
            )
            return search_result.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in web search for query '{query}': {str(e)}")
            return f"Error performing web search: {str(e)}"

    async def show_players(self, channel_id: int) -> List[str]:
        """Show the list of available NBA players ranked by fantasy value"""
        # If we have a draft in this channel, show available players from draft state
        if channel_id in self.draft_states and self.draft_states[channel_id].is_active:
            # For draft mode, we only show names since we store only names in draft state
            draft_state = self.draft_states[channel_id]
            all_players = await self._get_players()
            players = {name: all_players.get(name, {}) for name in draft_state.available_players}
            prefix = "Available Players in Draft"
        else:
            # Get players from cache or fetch fresh data
            players = await self._get_players()
            prefix = "NBA Players Ranked by Fantasy Value"
            
        if not players:
            return ["Could not fetch player rankings. Please try again later."]
        
        # Build the full content first
        content = []
        
        # If we have stats for players, create a formatted table
        first_player_stats = next(iter(players.values()))
        if first_player_stats:  # If we have stats
            # Most important columns for fantasy basketball
            essential_columns = [
                'R#',        # Rank
                'PLAYER',    # Player Name
                'TEAM',      # Team
                'POS',       # Position
                'PTS',       # Points
                'REB',       # Rebounds
                'AST',       # Assists
                'STL',       # Steals
                'BLK',       # Blocks
                'TO',        # Turnovers
                'FG%',       # Field Goal %
                '3PM',       # Three Pointers Made
                'FT%'        # Free Throw %
            ]
            
            # Filter columns that exist in our data
            filtered_columns = [col for col in essential_columns if col in first_player_stats]
            
            # Define column widths
            col_widths = {
                'rank': 4,   # Width for rank number (increased to accommodate the period)
                'name': max(max(len(name) for name in players.keys()), len("PLAYER")) + 1,  # Width for player names
            }
            
            # Calculate width for each stat column
            for col in filtered_columns:
                if col not in ['R#', 'PLAYER']:  # Skip rank and player name as they're handled separately
                    col_widths[col] = max(
                        max(len(str(stats.get(col, ''))) for stats in players.values()),  # Max width of values
                        len(col)  # Width of header
                    ) + 1  # Add minimal padding
            
            # Create header line
            header_line = f"{'#':>{col_widths['rank']}} {'PLAYER':<{col_widths['name']}}"
            for col in filtered_columns:
                if col not in ['R#', 'PLAYER']:
                    header_line += f" {col:^{col_widths[col]}}"
            
            # Create separator line
            separator = "-" * len(header_line)
            
            # Add header to content
            content.append(f"{prefix} (top {len(players)}):\n")
            content.append(header_line)
            content.append(separator)
            
            # Add player rows
            sorted_players = sorted(players.items(), key=lambda x: int(x[1].get('R#', '999')))
            for i, (player_name, stats) in enumerate(sorted_players, 1):
                player_line = f"{i:>2}. {player_name:<{col_widths['name']}}"  # Added period after rank
                for col in filtered_columns:
                    if col not in ['R#', 'PLAYER']:
                        value = stats.get(col, '')
                        # Right-align numeric columns, center-align others
                        if col in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', '3PM']:
                            player_line += f" {value:>{col_widths[col]}}"
                        else:
                            player_line += f" {value:^{col_widths[col]}}"
                content.append(player_line)
        else:
            # Simple numbered list for draft mode
            content.append(f"{prefix} (top {len(players)}):\n")
            for i, player_name in enumerate(players.keys(), 1):
                content.append(f"{i:>3}. {player_name}")
        
        # Join all content with newlines and split into messages
        return self._split_into_messages('\n'.join(content))

    async def show_my_team(self, channel_id: int) -> List[str]:
        """Show the user's current team in the draft"""
        if channel_id not in self.draft_states:
            return ["No draft has been started in this channel. Use !draft to start a new draft."]
            
        draft_state = self.draft_states[channel_id]
        if not draft_state.is_active:
            return ["The draft has ended. Start a new draft with !draft to build a new team."]
            
        if not draft_state.my_team:
            return ["Your team is currently empty. Wait for your turn to draft or use !pick to record your picks."]
            
        # Get the latest player stats
        all_players = await self._get_players()
        
        content = []
        content.append("🏀 Your Current Team 🏀\n")
        
        # Group players by position
        players_by_position = defaultdict(list)
        for player, position in draft_state.my_team:
            players_by_position[position if position else "FLEX"].append(player)
            
        # Display team composition
        for position in Position:
            players = players_by_position.get(position, [])
            if players:
                content.append(f"{position.value}:")
                for player in players:
                    # Use _find_player_match to get the correct player stats
                    match = self._find_player_match(player, all_players)
                    if match:
                        player_name, stats = match
                        # Use uppercase keys consistently
                        content.append(f"  • {player_name} - {stats.get('TEAM', 'N/A')} ({stats.get('POS', 'N/A')})")
                    else:
                        content.append(f"  • {player}")
                content.append("")
        
        # Show flex/unassigned players
        flex_players = players_by_position.get("FLEX", [])
        if flex_players:
            content.append("Unassigned Players:")
            for player in flex_players:
                # Use _find_player_match to get the correct player stats
                match = self._find_player_match(player, all_players)
                if match:
                    player_name, stats = match
                    # Use uppercase keys consistently
                    content.append(f"  • {player_name} - {stats.get('TEAM', 'N/A')} ({stats.get('POS', 'N/A')})")
                else:
                    content.append(f"  • {player}")
            content.append("")
        
        # Add draft status
        content.append(f"Draft Status:")
        content.append(f"• Round: {draft_state.current_round}/{draft_state.total_rounds}")
        content.append(f"• Your Position: Pick {draft_state.pick_position} of {draft_state.total_players}")
        content.append(f"• Total Players Drafted: {len(draft_state.my_team)}")
        
        return self._split_into_messages('\n'.join(content))

    async def _scrape_real_time_news(self, player_name: str) -> Optional[dict]:
        """Get real-time news using NBA stats API and Basketball Reference."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://www.nba.com',
            'Referer': 'https://www.nba.com/'
        }
        
        try:
            # Format player name for Basketball Reference URL
            # Example: "Nikola Jokic" -> "jokicni01"
            last_name = player_name.split()[-1].lower()
            first_name = player_name.split()[0].lower()
            bref_id = f"{last_name[:5]}{first_name[:2]}01"
            
            # URLs for different data sources
            bref_url = f"https://www.basketball-reference.com/players/{last_name[0]}/{bref_id}.html"
            
            async with aiohttp.ClientSession() as session:
                # Get latest game data from Basketball Reference
                async with session.get(bref_url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Get most recent game
                        game_log = soup.find('div', id='div_pgl_basic')
                        if game_log:
                            latest_game = game_log.find('tr')  # First row is most recent
                            if latest_game:
                                date = latest_game.find('td', {'data-stat': 'date_game'})
                                pts = latest_game.find('td', {'data-stat': 'pts'})
                                reb = latest_game.find('td', {'data-stat': 'trb'})
                                ast = latest_game.find('td', {'data-stat': 'ast'})
                                opp = latest_game.find('td', {'data-stat': 'opp_id'})
                                
                                if all([date, pts, reb, ast, opp]):
                                    return {
                                        'source': 'Basketball Reference',
                                        'date': date.text,
                                        'type': 'Game Performance',
                                        'headline': f"{player_name} vs {opp.text}",
                                        'description': f"Latest Game Stats: {pts.text} PTS, {reb.text} REB, {ast.text} AST"
                                    }
                
                # If we can't get the latest game, try to get their next game
                schedule_url = "https://www.basketball-reference.com/leagues/NBA_2025_games.html"
                async with session.get(schedule_url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find next game involving the player's team
                        schedule = soup.find('div', id='div_schedule')
                        if schedule:
                            upcoming_games = schedule.find_all('tr')
                            for game in upcoming_games:
                                if last_name.lower() in game.text.lower():
                                    date = game.find('th', {'data-stat': 'date_game'})
                                    visitor = game.find('td', {'data-stat': 'visitor_team_name'})
                                    home = game.find('td', {'data-stat': 'home_team_name'})
                                    if all([date, visitor, home]):
                                        return {
                                            'source': 'Basketball Reference',
                                            'date': date.text,
                                            'type': 'Upcoming Game',
                                            'headline': f"{player_name}'s Next Game",
                                            'description': f"{visitor.text} @ {home.text}"
                                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting stats for {player_name}: {str(e)}")
            return None

    async def _get_espn_player_id(self, player_name: str) -> Optional[str]:
        """Find ESPN player ID by searching their site."""
        try:
            search_url = f"https://www.espn.com/nba/players/_/search/{player_name.replace(' ', '+')}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find player link which contains the ID
                        player_link = soup.find('a', href=lambda x: x and '/nba/player/_/id/' in x)
                        if player_link:
                            player_id = player_link['href'].split('/id/')[1].split('/')[0]
                            return player_id
            return None
        except Exception as e:
            logger.error(f"Error finding ESPN ID for {player_name}: {str(e)}")
            return None

    async def get_player_news(self, player_name: str) -> List[str]:
        """Get the latest news about an NBA player from ESPN."""
        try:
            # First try to find the exact player name from our rankings
            players = await self._get_players()
            match = self._find_player_match(player_name, players)
            if match:
                player_name = match[0]  # Use the exact name from rankings
            
            # Look up player ID in our mapping
            player_id = None
            for known_name, espn_id in self.ESPN_PLAYER_IDS.items():
                if player_name.lower() in known_name.lower() or known_name.lower() in player_name.lower():
                    player_id = espn_id
                    player_name = known_name  # Use the exact name from our mapping
                    break
            
            if not player_id:
                return [f"Sorry, I don't have the ESPN ID for {player_name} in my database yet. Please try another player."]
            
            # Scrape ESPN player page
            url = f"https://www.espn.com/nba/player/_/id/{player_id}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        return [f"Error accessing ESPN page for {player_name}"]
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find the Fantasy Overview section which contains news
                    fantasy_news = soup.find('div', class_='FantasyOverview__News')
                    if fantasy_news:
                        # Get the News section
                        news_p = fantasy_news.find('p', class_='nws')
                        news_time = news_p.find('span', class_='FantasyNews__relDate') if news_p else None
                        news_content = news_p.find('span', class_='FantasyNews__content') if news_p else None
                        
                        # Get the Spin section
                        spin_p = fantasy_news.find('p', class_='spn')
                        spin_content = spin_p.find('span') if spin_p else None
                        
                        if news_content or spin_content:
                            response = [f"📰 Latest {player_name} Update 📰"]
                            
                            if news_time and news_content:
                                response.append(f"News ({news_time.text}): {news_content.text}")
                            
                            if spin_content:
                                response.append(f"\nSpin: {spin_content.text}")
                            
                            return ["\n".join(response)]
                    
                    return [f"No recent news found for {player_name} on ESPN"]
                
        except Exception as e:
            logger.error(f"Error getting news for {player_name}: {str(e)}")
            return [f"Error retrieving news for {player_name}. Please try again later."]
