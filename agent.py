import os
from collections import defaultdict
from typing import List, Dict
from mistralai import Mistral
import discord

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

class MistralAgent:
    def __init__(self):
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        # Dictionary to store conversation history per channel
        self.channel_history: Dict[int, List[dict]] = defaultdict(list)

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
