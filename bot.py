import os
import discord
import logging

from discord.ext import commands
from agent import MistralAgent
from dotenv import load_dotenv
PREFIX = "!"

# Setup logging
logger = logging.getLogger("discord")

class CustomHelpCommand(commands.HelpCommand):
    """Custom help command implementation"""
    
    async def send_bot_help(self, mapping):
        """Override the main help command"""
        embed = discord.Embed(
            title="🏀 Fantasy Basketball Assistant Commands",
            description="Here are all available commands:",
            color=discord.Color.blue()
        )
        
        # Add command fields to embed
        embed.add_field(
            name="📊 Draft Commands",
            value="```!draft - Start a new draft session\n!pick - Record a draft pick\n!getrec - Get draft recommendations\n!myteam - View your team\n!players - Show available players```",
            inline=False
        )
        
        embed.add_field(
            name="🔍 Player Analysis",
            value="```!compare - Compare multiple players\n!news - Get latest player updates```",
            inline=False
        )
        
        embed.add_field(
            name="ℹ️ Help",
            value="Type `!help <command>` for detailed information about a specific command",
            inline=False
        )
        
        # Send the embed to the channel
        await self.get_destination().send(embed=embed)
    
    async def send_command_help(self, command):
        """Keep the detailed help for individual commands"""
        embed = discord.Embed(
            title=f"!{command.name}",
            description=command.help or "No description available.",
            color=discord.Color.blue()
        )
        
        # Add usage field if command has usage info
        if command.usage:
            embed.add_field(name="Usage", value=f"```{command.usage}```", inline=False)
            
        await self.get_destination().send(embed=embed)

# Load the environment variables
load_dotenv()

# Create the bot with all intents and custom help command
intents = discord.Intents.all()
bot = commands.Bot(
    command_prefix=PREFIX,
    intents=intents,
    help_command=CustomHelpCommand()
)

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
    Only processes command messages starting with the prefix.
    """
    # Don't delete this line! It's necessary for the bot to process commands.
    await bot.process_commands(message)

    # Ignore all other messages
    return


# Commands
@bot.command(
    name="compare",
    help="🔄 Compare NBA players for fantasy value\n\n"
         "Description:\n"
         "• Compare multiple players' stats and value\n"
         "• Analyze current injuries and performance\n"
         "• Get detailed rankings and comparisons\n\n"
         "Usage: `!compare player1 player2 [player3 ...]`\n"
         "Example: `!compare \"LeBron James\" \"Stephen Curry\"`"
)
async def compare(ctx, *players):
    """Compare NBA players for fantasy basketball purposes."""
    # Check if we have at least 2 players to compare
    if len(players) < 2:
        await ctx.send(">>> Please provide at least 2 players to compare. Usage: `!compare player1 player2 [player3 ...]`")
        return
        
    logger.info(f"Comparing players: {', '.join(players)}")
    await ctx.send(">>> ⌛ Please wait 20 seconds-2 minutes while I analyze these players thoroughly...")
    
    response = await agent.compare_players(list(players))
    await ctx.send(response)


@bot.command(
    name="draft",
    help="🎮 Start a fantasy basketball draft\n\n"
         "Description:\n"
         "• Initialize new draft session\n"
         "• Set custom rounds and team count\n"
         "• Configure your draft position\n\n"
         "Usage: `!draft rounds pick_position total_picks`\n"
         "Example: `!draft 13 1 12`"
)
async def draft(ctx, rounds: int, pick_position: int, total_picks: int):
    """Start a fantasy basketball draft session."""
    if rounds < 1 or pick_position < 1 or total_picks < 2 or pick_position > total_picks:
        await ctx.send(">>> Invalid draft parameters! Please ensure:\n"
                      "- Rounds is at least 1\n"
                      "- Pick position is between 1 and total picks\n"
                      "- Total picks is at least 2")
        return

    response = await agent.start_draft(ctx.channel.id, rounds, pick_position, total_picks)
    await ctx.send(response)


@bot.command(
    name="pick",
    help="✏️ Record a draft pick\n\n"
         "Description:\n"
         "• Record player selections\n"
         "• Assign positions to your picks\n"
         "• Track draft progress\n\n"
         "Usage: `!pick number \"player_name\" [position]`\n"
         "Example: `!pick 1 \"LeBron James\" SF`"
)
async def pick(ctx, pick_num: int, player_name: str, *args):
    """Record a draft pick."""
    # Check if draft exists and is active
    if not hasattr(agent, 'draft_states') or ctx.channel.id not in agent.draft_states or not agent.draft_states[ctx.channel.id].is_active:
        await ctx.send(">>> No active draft in this channel! Start a draft first using the `!draft` command.")
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
                await ctx.send(">>> Invalid position! Please use: PG, SG, SF, PF, C, or UTIL")
                return
        player_name_parts = args[:-1]  # All but last argument is player name
    else:
        player_name_parts = []
    
    # Combine player name parts
    full_player_name = f"{player_name} {' '.join(player_name_parts)}".strip()
    
    response = await agent.update_draft_pick(ctx.channel.id, pick_num, full_player_name, position)
    await ctx.send(response)


@bot.command(
    name="getrec",
    help="💡 Get draft recommendations\n\n"
         "Description:\n"
         "• Get best available players\n"
         "• Receive strategic advice\n"
         "• Analyze team needs\n\n"
         "Usage: `!getrec`"
)
async def getrec(ctx):
    """Get draft recommendations considering draft position and current state."""
    # Check if draft exists and is active
    if not hasattr(agent, 'draft_states') or ctx.channel.id not in agent.draft_states or not agent.draft_states[ctx.channel.id].is_active:
        await ctx.send(">>> No active draft in this channel! Start a draft first using the `!draft` command.")
        return
        
    responses = await agent.get_draft_recommendation(ctx.channel.id)
    for response in responses:
        await ctx.send(response)


@bot.command(
    name="players",
    help="📊 Show available players\n\n"
         "Description:\n"
         "• View all available players\n"
         "• See fantasy rankings\n"
         "• Check player statistics\n\n"
         "Usage: `!players`"
)
async def players(ctx):
    """Show the list of available NBA players."""
    responses = await agent.show_players(ctx.channel.id)
    for response in responses:
        await ctx.send(response)


@bot.command(
    name="myteam",
    help="👥 View your current team\n\n"
         "Description:\n"
         "• See your drafted players\n"
         "• Check team composition\n"
         "• View roster by position\n\n"
         "Usage: `!myteam`"
)
async def myteam(ctx):
    """Show your current team in the draft."""
    responses = await agent.show_my_team(ctx.channel.id)
    for response in responses:
        await ctx.send(response)


@bot.command(
    name="news",
    help="📰 Get player news and updates\n\n"
         "Description:\n"
         "• Check injury status\n"
         "• See recent performance\n"
         "• Get latest updates\n\n"
         "Usage: `!news \"player_name\"`\n"
         "Example: `!news \"LeBron James\"`"
)
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
