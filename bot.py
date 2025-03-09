import os
import discord
import logging

from discord.ext import commands
from agent import MistralAgent
from dotenv import load_dotenv
PREFIX = "!"

# Setup logging
logger = logging.getLogger("discord")

# Load the environment variables
load_dotenv()

# Create the bot with all intents
# The message content and members intent must be enabled in the Discord Developer Portal for the bot to work.
intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# Import the Mistral agent from the agent.py file
agent = MistralAgent()


# Get the token from the environment variables
token = os.getenv("DISCORD_TOKEN")


@bot.event
async def on_ready():
    """
    Called when the client is done preparing the data received from Discord.
    Prints message on terminal when bot successfully connects to discord.

    https://discordpy.readthedocs.io/en/latest/api.html#discord.on_ready
    """
    logger.info(f"{bot.user} has connected to Discord!")


@bot.event
async def on_message(message: discord.Message):
    """
    Called when a message is sent in any channel the bot can see.

    https://discordpy.readthedocs.io/en/latest/api.html#discord.on_message
    """
    # Don't delete this line! It's necessary for the bot to process commands.
    await bot.process_commands(message)

    # Ignore messages from self or other bots to prevent infinite loops.
    if message.author.bot or message.content.startswith("!"):
        return

    # Process the message with the agent you wrote
    # Open up the agent.py file to customize the agent
    logger.info(f"Processing message from {message.author}: {message.content}")
    response = await agent.run(message)

    # Send the response back to the channel
    await message.reply(response)


# Commands
@bot.command(
    name="compare",
    help="""Compare NBA players for fantasy basketball.
Arguments:
• player1, player2, ... - Names of players to compare (minimum 2 players)

Usage: !compare player1 player2 [player3 ...]
Example: !compare "LeBron James" "Stephen Curry" "Kevin Durant\""""
)
async def compare(ctx, *players):
    """Compare NBA players for fantasy basketball purposes."""
    # Check if we have at least 2 players to compare
    if len(players) < 2:
        await ctx.send("Please provide at least 2 players to compare. Usage: !compare player1 player2 [player3 ...]")
        return

    logger.info(f"Comparing players: {', '.join(players)}")
    response = await agent.compare_players(list(players))
    await ctx.send(response)


@bot.command(
    name="draft",
    help="""Start a fantasy basketball draft.
Arguments:
• rounds - Number of rounds in the draft (e.g. 13)
• pick_position - Your position in the draft order (e.g. 1 for first pick)
• total_picks - Total number of teams drafting (e.g. 12 for a 12-team league)

Usage: !draft rounds pick_position total_picks"""
)
async def draft(ctx, rounds: int, pick_position: int, total_picks: int):
    """Start a fantasy basketball draft session."""
    if rounds < 1 or pick_position < 1 or total_picks < 2 or pick_position > total_picks:
        await ctx.send("Invalid draft parameters! Please ensure:\n"
                      "- Rounds is at least 1\n"
                      "- Pick position is between 1 and total picks\n"
                      "- Total picks is at least 2")
        return

    response = await agent.start_draft(ctx.channel.id, rounds, pick_position, total_picks)
    await ctx.send(response)


@bot.command(
    name="pick",
    help="""Record a draft pick.
Arguments:
• pick_number - The overall pick number in the draft (e.g. 1 for first pick)
• player_name - Name of the player being drafted (e.g. "LeBron James" or "LeBron")
• position - REQUIRED when it's your turn to draft. Must be one of: PG/SG/SF/PF/C/UTIL

Usage when it's your turn:
!pick <pick_number> <player_name> <position>

Usage for other picks:
!pick <pick_number> <player_name> [position]""",
    
)
async def pick(ctx, pick_num: int, player_name: str, *args):
    """Record a draft pick. Position (PG/SG/SF/PF/C/UTIL) is required when it's your turn to assign the player to your lineup."""
    # Check if draft exists and is active
    if not hasattr(agent, 'draft_states') or ctx.channel.id not in agent.draft_states or not agent.draft_states[ctx.channel.id].is_active:
        await ctx.send("No active draft in this channel! Start a draft first using the !draft command.")
        return
    
    draft_state = agent.draft_states[ctx.channel.id]
    current_pick = (draft_state.picks_made % draft_state.total_players) + 1
    
    # Combine all remaining args into player name and position
    position = None
    if args:
        # If it's the user's pick, the last argument must be position
        if current_pick == draft_state.pick_position:
            position = args[-1].upper()
            if position not in ["PG", "SG", "SF", "PF", "C", "UTIL"]:
                await ctx.send("Invalid position! Please use: PG, SG, SF, PF, C, or UTIL")
                return
            player_name_parts = args[:-1]  # All but last argument is player name
        else:
            # For non-user picks, check if last arg is a valid position
            last_arg = args[-1].upper()
            if last_arg in ["PG", "SG", "SF", "PF", "C", "UTIL"]:
                position = last_arg
                player_name_parts = args[:-1]
            else:
                # All args are part of the player name
                player_name_parts = args
    else:
        player_name_parts = []
    
    # Combine player name parts
    full_player_name = f"{player_name} {' '.join(player_name_parts)}".strip()
    
    response = await agent.update_draft_pick(ctx.channel.id, pick_num, full_player_name, position)
    await ctx.send(response)


@bot.command(
    name="getrec",
    help="""Get draft recommendations based on your draft position and current state.
Shows:
• Draft position analysis
• Draft strategy
• Top available players
• Best picks for your position

Usage: !getrec"""
)
async def getrec(ctx):
    """Get draft recommendations considering draft position and current state."""
    # Check if draft exists and is active
    if not hasattr(agent, 'draft_states') or ctx.channel.id not in agent.draft_states or not agent.draft_states[ctx.channel.id].is_active:
        await ctx.send("No active draft in this channel! Start a draft first using the !draft command.")
        return
        
    responses = await agent.get_draft_recommendation(ctx.channel.id)
    for response in responses:
        await ctx.send(response)


@bot.command(
    name="players",
    help="""Show the list of available NBA players ranked by fantasy value.
During draft: Shows only undrafted players
Outside draft: Shows all NBA players with their stats

Usage: !players"""
)
async def players(ctx):
    """Show the list of available NBA players."""
    responses = await agent.show_players(ctx.channel.id)
    for response in responses:
        await ctx.send(response)


@bot.command(
    name="myteam",
    help="""Show your current team in the draft.
Displays:
• Players grouped by position
• Team stats and information
• Draft status and progress

Usage: !myteam"""
)
async def myteam(ctx):
    """Show your current team in the draft."""
    responses = await agent.show_my_team(ctx.channel.id)
    for response in responses:
        await ctx.send(response)


@bot.command(name="news", help="Get the latest news about an NBA player. Usage: !news player_name")
async def news(ctx, player_name: str, *, rest: str = ""):
    """Get recent news and updates about an NBA player."""
    # Combine player name if it was split across arguments
    full_player_name = f"{player_name} {rest}".strip()
    
    logger.info(f"Fetching news for player: {full_player_name}")
    responses = await agent.get_player_news(full_player_name)
    for response in responses:
        await ctx.send(response)


# Start the bot, connecting it to the gateway
bot.run(token)
