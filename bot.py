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
@bot.command(name="compare", help="Compare NBA players for fantasy basketball. Usage: !compare player1 player2 [player3 ...]")
async def compare(ctx, *players):
    """Compare NBA players for fantasy basketball purposes."""
    # Check if we have at least 2 players to compare
    if len(players) < 2:
        await ctx.send("Please provide at least 2 players to compare. Usage: !compare player1 player2 [player3 ...]")
        return

    logger.info(f"Comparing players: {', '.join(players)}")
    response = await agent.compare_players(list(players))
    await ctx.send(response)


@bot.command(name="draft", help="Start a fantasy basketball draft. Usage: !draft rounds pick_position total_picks")
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


@bot.command(name="pick", help="Record a draft pick. Usage: !pick player_name position")
async def pick(ctx, player_name: str, position: str, *, rest: str = ""):
    """Record a draft pick."""
    # Check if draft exists and is active
    if not hasattr(agent, 'draft_states') or ctx.channel.id not in agent.draft_states or not agent.draft_states[ctx.channel.id].is_active:
        await ctx.send("No active draft in this channel! Start a draft first using the !draft command.")
        return
    
    # Combine player name if it was split
    full_player_name = f"{player_name} {rest}".strip()
    
    response = await agent.update_draft_pick(ctx.channel.id, full_player_name, position)
    await ctx.send(response)


@bot.command(name="myturn", help="Get draft recommendations when it's your turn")
async def myturn(ctx):
    """Get draft recommendations when it's your turn."""
    # Check if draft exists and is active
    if not hasattr(agent, 'draft_states') or ctx.channel.id not in agent.draft_states or not agent.draft_states[ctx.channel.id].is_active:
        await ctx.send("No active draft in this channel! Start a draft first using the !draft command.")
        return
        
    response = await agent.get_draft_recommendation(ctx.channel.id)
    await ctx.send(response)


@bot.command(name="players", help="Show the list of available NBA players ranked by fantasy value")
async def players(ctx):
    """Show the list of available NBA players."""
    responses = await agent.show_players(ctx.channel.id)
    for response in responses:
        await ctx.send(response)


# This example command is here to show you how to add commands to the bot.
# Run !ping with any number of arguments to see the command in action.
# Feel free to delete this if your project will not need commands.
@bot.command(name="ping", help="Pings the bot.")
async def ping(ctx, *, arg=None):
    if arg is None:
        await ctx.send("Pong!")
    else:
        await ctx.send(f"Pong! Your argument was {arg}")


# Start the bot, connecting it to the gateway
bot.run(token)
