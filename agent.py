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
        current_pick_in_round = (self.picks_made % self.total_players) + 1
        return current_pick_in_round == self.pick_position
    
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

    async def compare_players(self, players: List[str]) -> str:
        """Compare NBA players for fantasy basketball purposes without maintaining conversation history."""
        # Format the players list for the prompt
        players_str = ", ".join(players)
        
        # Get current information about all players in one search
        search_term = f"{players_str} NBA injury status fantasy basketball current 2024"
        explanation = f"Searching for current NBA information about these players: {players_str}"
        
        # Use web_search tool to get current information
        web_search_result = "<function_calls>\n<invoke name=\"web_search\">\n<parameter name=\"search_term\">" + search_term + "</parameter>\n<parameter name=\"explanation\">" + explanation + "</parameter>\n</invoke>\n</function_calls>"
        
        messages = [
            {"role": "system", "content": COMPARE_PROMPT.format(
                players=players_str,
                current_info=web_search_result
            )},
            {"role": "user", "content": f"Compare these players for fantasy basketball: {players_str}"}
        ]

        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )

        return response.choices[0].message.content

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
            draft_state.available_players = await self.fetch_players_list()
            
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
• !pick <player_name> <position> - Record a draft pick
• !myturn - Get recommendations for your pick
• !myteam - View your current roster
• !players - View available players

Valid Positions: PG, SG, SF, PF, C, G, F, UTIL

Your pick will come up every {total_players} picks. Good luck! 🎯"""
            
        except Exception as e:
            logger.error(f"Error starting draft: {str(e)}")
            return f"Error starting draft: {str(e)}\nPlease try again or contact support if the issue persists."

    async def update_draft_pick(self, channel_id: int, drafted_player: str, position_str: str) -> str:
        """Update draft state with a new pick"""
        if channel_id not in self.draft_states or not self.draft_states[channel_id].is_active:
            return "No active draft in this channel!"
        
        try:
            position = Position[position_str.upper()]
        except KeyError:
            return f"Invalid position! Please use one of: {', '.join([pos.name for pos in Position])}"
        
        draft_state = self.draft_states[channel_id]
        draft_state.update_draft(drafted_player, position)
        
        if not draft_state.is_active:
            # Generate draft summary when complete
            my_team = draft_state.my_team
            summary = "🎉 Draft Complete! 🎉\n\nYour Final Roster:\n"
            for pos in Position:
                players = [player for player, p_pos in my_team if p_pos == pos]
                if players:
                    summary += f"\n{pos.value}:"
                    for player in players:
                        summary += f"\n• {player}"
            return summary
        
        # If it was our pick, show updated roster needs
        if draft_state.is_user_turn():
            needs = draft_state.get_roster_needs()
            needs_str = "\nRoster Needs:"
            for pos, count in needs.items():
                if count > 0:
                    needs_str += f"\n• {pos.value}: {count} needed"
            
            picks_until_next = draft_state.total_players - (draft_state.picks_made % draft_state.total_players)
            
            return f"""✅ Pick Recorded: {drafted_player} ({position.value})

Draft Status:
• Round: {draft_state.current_round}/{draft_state.total_rounds}
• Pick: {(draft_state.picks_made % draft_state.total_players) + 1}/{draft_state.total_players}
• Players Remaining: {len(draft_state.available_players)}
• Picks until your next turn: {picks_until_next}{needs_str}"""
        
        return f"""✅ Pick Recorded: {drafted_player} ({position.value})

Draft Status:
• Round: {draft_state.current_round}/{draft_state.total_rounds}
• Pick: {(draft_state.picks_made % draft_state.total_players) + 1}/{draft_state.total_players}
• Players Remaining: {len(draft_state.available_players)}"""

    async def get_draft_recommendation(self, channel_id: int) -> str:
        """Get draft recommendation when it's user's turn"""
        if channel_id not in self.draft_states or not self.draft_states[channel_id].is_active:
            return "No active draft in this channel!"
        
        draft_state = self.draft_states[channel_id]
        if not draft_state.is_user_turn():
            return "It's not your turn to draft!"
        
        # Get roster needs
        needs = draft_state.get_roster_needs()
        needs_str = "Current roster needs:\n"
        for pos, count in needs.items():
            if count > 0:
                needs_str += f"• {pos.value}: {count} needed\n"
        
        # Get current information about available players
        available_players_str = ", ".join(draft_state.available_players[:10])
        
        # Use web search to get current information
        messages = [
            {"role": "system", "content": f"""You are a fantasy basketball draft expert. Consider:
1. Current round: {draft_state.current_round}/{draft_state.total_rounds}
2. Draft position: {draft_state.pick_position}/{draft_state.total_players}
3. My team so far: {', '.join([f"{player} ({pos.value})" for player, pos in draft_state.my_team])}
4. Roster needs:
{needs_str}

Recommend the best available player to draft based on:
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
        
        return f"Draft Analysis:\n{needs_str}\n{response.choices[0].message.content}"

    async def show_players(self, channel_id: int) -> List[str]:
        """Show the list of available NBA players ranked by fantasy value"""
        current_time = int(time.time())
        
        # If we have a draft in this channel, show available players from draft state
        if channel_id in self.draft_states and self.draft_states[channel_id].is_active:
            # For draft mode, we only show names since we store only names in draft state
            players = [(p, {}) for p in self.draft_states[channel_id].available_players]
            prefix = "Available Players in Draft"
        else:
            # Refresh cache if it's empty or older than 1 hour
            if not self.cached_players or (current_time - self.last_fetch_time) > 3600:
                self.cached_players = await self.fetch_players_list()
                self.last_fetch_time = current_time
            
            players = self.cached_players
            prefix = "NBA Players Ranked by Fantasy Value"
            
        if not players:
            return ["Could not fetch player rankings. Please try again later."]
        
        # Build the responses
        responses = []
        
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
            
            # Start first chunk with header
            current_chunk = f"{prefix} (top {len(players)}):\n\n{header_line}\n{separator}\n"
            
            # Add player rows
            for i, (player_name, stats) in enumerate(players, 1):
                player_line = f"{i:>{col_widths['rank']}} {player_name:<{col_widths['name']}}"
                for col in filtered_columns[2:]:  # Skip rank and name columns
                    value = stats.get(col, '')
                    player_line += f" {value:^{col_widths[col]}}"
                player_line += "\n"
                
                # If adding this line would exceed Discord's limit, start a new chunk
                if len(current_chunk) + len(player_line) > 1900:
                    responses.append(current_chunk)
                    current_chunk = ""  # Start new chunk without header
                
                current_chunk += player_line
        else:
            # Simple numbered list for draft mode
            current_chunk = f"{prefix} (top {len(players)}):\n\n"
            for i, (player_name, _) in enumerate(players, 1):
                player_line = f"{i:>3}. {player_name}\n"
                
                # If adding this line would exceed Discord's limit, start a new chunk
                if len(current_chunk) + len(player_line) > 1900:
                    responses.append(current_chunk)
                    current_chunk = ""
                
                current_chunk += player_line
        
        # Add the last chunk if it's not empty
        if current_chunk:
            responses.append(current_chunk)
            
        return responses
