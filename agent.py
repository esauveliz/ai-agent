import os
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from mistralai import Mistral
import discord
import aiohttp
import logging
from bs4 import BeautifulSoup
import time

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
        self.cached_players: List[str] = []  # Cache for player rankings
        self.last_fetch_time = 0  # Track when we last fetched players

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

    def _find_player_match(self, search_name: str, players_list: List[Tuple[str, Dict[str, str]]]) -> Optional[Tuple[str, Dict[str, str]]]:
        """Find the best match for a player name in the players list."""
        # Handle names without spaces (e.g., "JimmyButler" -> "Jimmy Butler")
        def split_name(name: str) -> str:
            # Common prefixes to handle (e.g., "Mc", "Mac", etc.)
            prefixes = ["Mc", "Mac", "De", "Van", "Von"]
            
            # First try to split on capital letters
            parts = []
            current = []
            for i, char in enumerate(name):
                if i > 0 and char.isupper():
                    # Check if it's part of a prefix
                    current_word = "".join(current)
                    next_chars = name[i:i+3] if i+3 <= len(name) else name[i:]  # Look ahead
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
        search_parts = search_name.lower().split()
        
        # First try exact match
        for player_name, stats in players_list:
            if player_name.lower() == search_name.lower():
                return (player_name, stats)
        
        # Then try first+last name match
        if len(search_parts) > 1:
            for player_name, stats in players_list:
                player_parts = player_name.lower().split()
                if len(player_parts) > 1:
                    # Match both first and last name
                    if (search_parts[0] in player_parts[0] and 
                        search_parts[-1] in player_parts[-1]):
                        return (player_name, stats)
        
        # Try matching without spaces
        no_space_search = "".join(search_parts)
        for player_name, stats in players_list:
            no_space_player = "".join(player_name.lower().split())
            if no_space_search in no_space_player:
                return (player_name, stats)
        
        # Finally try partial match on either first or last name
        for player_name, stats in players_list:
            player_parts = player_name.lower().split()
            if any(part in player_parts for part in search_parts):
                return (player_name, stats)
        
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

    async def _get_players(self, force_refresh: bool = False) -> List[Tuple[str, Dict[str, str]]]:
        """Get players from cache if available and fresh, otherwise fetch new data.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            List of tuples containing (player_name, stats_dict)
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

    async def fetch_players_list(self) -> List[Tuple[str, Dict[str, str]]]:
        """Fetch current NBA players list from hashtagbasketball.com"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        try:
            players = []
            url = "https://hashtagbasketball.com/fantasy-basketball-rankings"
            
            logger.info("Starting to fetch top 215 players from hashtagbasketball.com")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch players: {response.status}")
                        return []
                    
                    logger.info("Successfully got response from server")
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find the rankings table
                    rankings_table = soup.find('table', {'id': 'ContentPlaceHolder1_GridView1'})
                    
                    if not rankings_table:
                        logger.warning("Could not find rankings table")
                        return []
                    
                    # Get headers
                    header_row = rankings_table.find('tr')
                    if not header_row:
                        logger.warning("Could not find table headers")
                        return []
                    
                    headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
                    
                    # Extract player data from table rows
                    rows = rankings_table.find_all('tr')[1:]  # Skip header row
                    if not rows:
                        logger.warning("No player rows found in rankings table")
                        return []
                    
                    # Extract player data from table rows
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:  # Make sure we have enough cells
                            player_name = cells[1].get_text(strip=True)  # Player name is in second column
                            if player_name and player_name != "PLAYER":  # Skip entries that are just "PLAYER"
                                # Create dictionary of all stats
                                stats = {}
                                for i, cell in enumerate(cells):
                                    if i < len(headers):  # Make sure we have a header for this column
                                        stats[headers[i]] = cell.get_text(strip=True)
                                players.append((player_name, stats))
                    
                    if not players:
                        logger.warning("No players found in rankings")
                        return []
                    
                    initial_count = len(players)
                    # Filter out any remaining "PLAYER" entries
                    players = [p for p in players if p[0] != "PLAYER"]
                    if len(players) != initial_count:
                        logger.info(f"Removed {initial_count - len(players)} invalid player entries")
                    
                    logger.info(f"Successfully fetched {len(players)} players from hashtagbasketball.com")
                    return players  # Keep original order as it's already ranked by fantasy value
                    
        except Exception as e:
            logger.error(f"Error fetching players from hashtagbasketball.com: {str(e)}")
            return []

    async def start_draft(self, channel_id: int, total_rounds: int, pick_position: int, total_players: int) -> str:
        """Initialize a new draft session"""
        # Check if there's already an active draft
        if channel_id in self.draft_states and self.draft_states[channel_id].is_active:
            return "A draft is already in progress in this channel! Use !pick to record picks."
        
        draft_state = DraftState(total_rounds, pick_position, total_players)
        
        try:
            # Fetch initial player list
            players_with_stats = await self._get_players()
            draft_state.available_players = [player[0] for player in players_with_stats]
            
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
• Players Available: {len(draft_state.available_players)}

Available Commands:
• !pick <pick_number> <player_name> - Record which player was picked at each draft position
• !myturn - Get recommendations for your pick
• !myteam - View your current roster
• !players - View available players

Your pick will come up every {total_players} picks. Good luck! 🎯"""
            
        except Exception as e:
            logger.error(f"Error starting draft: {str(e)}")
            return f"Error starting draft: {str(e)}\nPlease try again or contact support if the issue persists."

    async def update_draft_pick(self, channel_id: int, pick_number: str, *player_name_parts: str) -> str:
        """Update draft state with a new pick"""
        if channel_id not in self.draft_states or not self.draft_states[channel_id].is_active:
            return "No active draft in this channel!"
        
        draft_state = self.draft_states[channel_id]
        
        try:
            # Convert pick number to integer and validate
            pick_num = int(pick_number)
            if pick_num < 1:
                return "Pick number must be positive!"
            
            # Join the player name parts
            player_search = " ".join(player_name_parts).lower()
            if not player_search:
                return "Please provide a player name!"
            
            # Find the best matching player in available players
            best_match = None
            for player in draft_state.available_players:
                if player_search in player.lower():
                    if best_match is None or len(player) < len(best_match):
                        best_match = player
                # Also check first/last name exact matches
                player_parts = player.lower().split()
                if player_search in player_parts:
                    best_match = player
                    break
            
            if not best_match:
                return f"Could not find player '{player_search}' in available players. Use !players to see the list of available players."
            
            # Remove the player from available players
            draft_state.available_players.remove(best_match)
            
            # Record the pick
            draft_state.picks_made += 1
            
            # Calculate if this pick corresponds to the user's position
            current_round = (draft_state.picks_made - 1) // draft_state.total_players + 1
            pick_in_round = ((draft_state.picks_made - 1) % draft_state.total_players) + 1
            is_user_pick = pick_in_round == draft_state.pick_position
            
            # Add to drafted players and user's team if it's their pick
            draft_state.drafted_players.append((best_match, None))
            if is_user_pick:
                draft_state.my_team.append((best_match, None))
            
            # Update round if necessary
            draft_state.current_round = current_round
            
            # Check if draft is complete
            if draft_state.picks_made >= (draft_state.total_players * draft_state.total_rounds):
                draft_state.is_active = False
                return "🎉 Draft Complete! 🎉"
            
            # If it's our pick, add to my team
            if draft_state.is_user_turn():
                picks_until_next = draft_state.total_players - (draft_state.picks_made % draft_state.total_players)
                
                return f"""✅ Pick #{pick_num}: {best_match}

Draft Status:
• Round: {draft_state.current_round}/{draft_state.total_rounds}
• Pick: {(draft_state.picks_made % draft_state.total_players) + 1}/{draft_state.total_players}
• Players Remaining: {len(draft_state.available_players)}
• Picks until your next turn: {picks_until_next}"""
            
            return f"""✅ Pick #{pick_num}: {best_match}

Draft Status:
• Round: {draft_state.current_round}/{draft_state.total_rounds}
• Pick: {(draft_state.picks_made % draft_state.total_players) + 1}/{draft_state.total_players}
• Players Remaining: {len(draft_state.available_players)}"""
            
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
        """Get draft recommendation when it's user's turn"""
        if channel_id not in self.draft_states or not self.draft_states[channel_id].is_active:
            return ["No active draft in this channel!"]
        
        draft_state = self.draft_states[channel_id]
        if not draft_state.is_user_turn():
            return ["It's not your turn to draft!"]
        
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
        explanation = f"Getting current information about top available players: {available_players_str}"
        web_search_result = "<function_calls>\n<invoke name=\"web_search\">\n<parameter name=\"search_term\">" + search_term + "</parameter>\n<parameter name=\"explanation\">" + explanation + "</parameter>\n</invoke>\n</function_calls>"
        
        messages = [
            {"role": "system", "content": f"""You are a fantasy basketball draft expert. Consider:
1. Current round: {draft_state.current_round}/{draft_state.total_rounds}
2. Draft position: {draft_state.pick_position}/{draft_state.total_players}
3. My team so far: {', '.join([f"{player}" for player, _ in draft_state.my_team])}
4. Roster needs:
{needs_str}

Current player information:
{web_search_result}

Provide draft recommendations in this format:
1. Draft Strategy Summary: 2-3 sentences about what to prioritize based on draft position, round, and team needs
2. Top 5 Available Players: List 5 best available players with 1-line explanation for each
3. Final Recommendation: State best pick and second-best pick with brief reasoning

Focus on:
- Team needs and roster construction
- Current fantasy value and projections
- Position scarcity
- Injury status and availability
- Recent performance and trends"""},
            {"role": "user", "content": f"Who should I draft from these available players: {available_players_str}?"}
        ]

        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )
        
        full_response = f"""Draft Analysis:
{needs_str}
{response.choices[0].message.content}"""
        return self._split_into_messages(full_response)

    async def web_search(self, query: str) -> str:
        """Perform a web search using the web_search tool."""
        try:
            # Use the web_search tool directly
            return f"<function_calls><invoke name=\"web_search\"><parameter name=\"search_term\">{query}</parameter><parameter name=\"explanation\">Getting current NBA player information</parameter></invoke></function_calls>"
        except Exception as e:
            return f"Error performing web search: {str(e)}"

    async def show_players(self, channel_id: int) -> List[str]:
        """Show the list of available NBA players ranked by fantasy value"""
        # If we have a draft in this channel, show available players from draft state
        if channel_id in self.draft_states and self.draft_states[channel_id].is_active:
            # For draft mode, we only show names since we store only names in draft state
            players = [(p, {}) for p in self.draft_states[channel_id].available_players]
            prefix = "Available Players in Draft"
        else:
            # Get players from cache or fetch fresh data
            players = await self._get_players()
            prefix = "NBA Players Ranked by Fantasy Value"
            
        if not players:
            return ["Could not fetch player rankings. Please try again later."]
        
        # Build the full content first
        content = []
        
        # If we have stats, create a formatted table
        if players[0][1]:  # If we have stats for the first player
            # Get column widths
            columns = list(players[0][1].keys())
            # Remove unwanted columns
            columns_to_exclude = {'GP', 'MPG', 'FT%', '3PM'}
            filtered_columns = [col for col in columns if col not in columns_to_exclude]
            
            col_widths = {
                'rank': 4,  # Width for rank number
                'name': max(max(len(p[0]) for p in players), len("Name")) + 2,  # Width for player names
            }
            
            # Calculate width for each stat column
            for col in filtered_columns[2:]:  # Skip rank and name columns
                col_widths[col] = max(
                    max(len(p[1].get(col, '')) for p in players),  # Max width of values
                    len(col)  # Width of header
                ) + 2  # Add padding
            
            # Create header line
            header_line = f"{'#':>{col_widths['rank']}} {'Name':<{col_widths['name']}}"
            for col in filtered_columns[2:]:  # Skip rank and name columns
                header_line += f" {col:^{col_widths[col]}}"
            
            # Create separator line
            separator = "=" * len(header_line)
            
            # Add header to content
            content.append(f"{prefix} (top {len(players)}):\n")
            content.append(header_line)
            content.append(separator)
            
            # Add player rows
            for i, (player_name, stats) in enumerate(players, 1):
                player_line = f"{i:>{col_widths['rank']}} {player_name:<{col_widths['name']}}"
                for col in filtered_columns[2:]:  # Skip rank and name columns
                    value = stats.get(col, '')
                    player_line += f" {value:^{col_widths[col]}}"
                content.append(player_line)
        else:
            # Simple numbered list for draft mode
            content.append(f"{prefix} (top {len(players)}):\n")
            for i, (player_name, _) in enumerate(players, 1):
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
        player_stats = {p[0]: p[1] for p in all_players}
        
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
                    stats = player_stats.get(player, {})
                    if stats:
                        # Show key stats if available
                        content.append(f"  • {player} - {stats.get('Team', 'N/A')} ({stats.get('Pos', 'N/A')})")
                    else:
                        content.append(f"  • {player}")
                content.append("")
        
        # Show flex/unassigned players
        flex_players = players_by_position.get("FLEX", [])
        if flex_players:
            content.append("Unassigned Players:")
            for player in flex_players:
                stats = player_stats.get(player, {})
                if stats:
                    content.append(f"  • {player} - {stats.get('Team', 'N/A')} ({stats.get('Pos', 'N/A')})")
                else:
                    content.append(f"  • {player}")
            content.append("")
        
        # Add draft status
        content.append(f"Draft Status:")
        content.append(f"• Round: {draft_state.current_round}/{draft_state.total_rounds}")
        content.append(f"• Your Position: Pick {draft_state.pick_position} of {draft_state.total_players}")
        content.append(f"• Total Players Drafted: {len(draft_state.my_team)}")
        
        return self._split_into_messages('\n'.join(content))
