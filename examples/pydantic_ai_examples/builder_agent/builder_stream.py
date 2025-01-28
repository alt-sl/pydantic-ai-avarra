"""Builder Agent Implementation with Simulated Streaming

This module implements a Builder Agent that can design and configure other agents based on
natural language descriptions, with simulated streaming output for better UX.
"""

from dataclasses import dataclass
import asyncio
from typing import Optional, Literal, Union, List

from pydantic import BaseModel, Field
from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.prompt import Prompt

from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.messages import ModelMessage

# Initialize rich console for pretty output
console = Console()

class AgentConfig(BaseModel):
    """Configuration for a new agent to be created."""
    
    system_prompt: str = Field(
        description="The system prompt that defines the agent's role and behavior"
    )
    model_name: Literal[
        'claude-3-5-sonnet-latest'
    ] = Field(
        default='claude-3-5-sonnet-latest',
        description="The model to use for this agent"
    )
    name: str = Field(
        description="A descriptive name for the agent"
    )
    description: str = Field(
        description="A brief description of what this agent does"
    )

class AgentResponse(BaseModel):
    """Response from the created agent."""
    response: str = Field(description="The agent's response to the query")

@dataclass
class BuilderDeps:
    """Dependencies for the Builder Agent.
    
    This maintains state about the currently created agent and its configuration.
    """
    message_history: List[ModelMessage]
    current_config: Optional[AgentConfig] = None
    created_agent: Optional[Agent[None, str]] = None

    def __init__(self):
        self.message_history = []

# The Builder Agent itself
builder_agent = Agent[BuilderDeps, Union[AgentConfig, AgentResponse]](
    'claude-3-5-sonnet-latest',
    deps_type=BuilderDeps,
    result_type=Union[AgentConfig, AgentResponse],
    system_prompt="""You are an expert at designing AI agents. Your job is to:

1. Understand what kind of agent the user wants based on their description
2. Design an appropriate system prompt that will make the agent behave as desired
3. Choose an appropriate model for the agent's needs
4. Create a clear name and description for the agent

When designing system prompts:
- Make them clear and specific
- Include any necessary constraints or guidelines
- Define the agent's role and capabilities
- Set the appropriate tone and style

We will use claude-3-5-sonnet-latest for all agents as it provides excellent capabilities for:
- Complex reasoning and analysis
- Creative writing and generation
- Technical understanding
- Consistent and high-quality outputs

Always validate that your configurations make sense for the user's needs."""
)

@builder_agent.tool
async def create_agent(ctx: RunContext[BuilderDeps], config: AgentConfig) -> AgentConfig:
    """Create a new agent with the given configuration."""
    ctx.deps.current_config = config
    new_agent = Agent[None, str](
        config.model_name,
        system_prompt=config.system_prompt,
        name=config.name,
    )
    ctx.deps.created_agent = new_agent
    return config

@builder_agent.tool
async def handoff_to_agent(ctx: RunContext[BuilderDeps], query: str) -> AgentResponse:
    """Hand off the conversation to the created agent."""
    if not ctx.deps.created_agent:
        raise ModelRetry("No agent has been created yet. Please create an agent first.")
    
    result = await ctx.deps.created_agent.run(query, message_history=ctx.deps.message_history)
    ctx.deps.message_history.extend(result.all_messages())
    return AgentResponse(response=result.data)

async def stream_text(text: str, delay: float = 0.02):
    """Simulate streaming text output."""
    with Live(Text(""), refresh_per_second=20) as live:
        current_text = ""
        for char in text:
            current_text += char
            live.update(Text(current_text))
            await asyncio.sleep(delay)

async def main():
    """Example usage of the Builder Agent with streaming output."""
    deps = BuilderDeps()
    
    while True:
        request = Prompt.ask(
            'What kind of agent would you like to create? (or "quit" to exit)',
            default="quit"
        )
        
        if request.lower() == "quit":
            break

        # Create the agent based on the description
        result = await builder_agent.run(request, deps=deps)
        
        if isinstance(result.data, AgentConfig):
            console.print("\n[bold blue]Created agent configuration:[/]")
            await stream_text(f"Name: {result.data.name}")
            await stream_text(f"\nDescription: {result.data.description}")
            await stream_text(f"\nModel: {result.data.model_name}")
            await stream_text(f"\nSystem Prompt: {result.data.system_prompt}")
            console.print("\n[bold green]Configuration complete![/]")
        
            # Now we can interact with the created agent
            while True:
                query = Prompt.ask(
                    '\nWhat would you like to ask the agent? (or "back" to create a new agent)',
                    default="back"
                )
                
                if query.lower() == "back":
                    break
                    
                # Get the agent's response with streaming simulation
                result = await builder_agent.run(query, deps=deps)
                if isinstance(result.data, AgentResponse):
                    console.print("\n[bold cyan]Agent response:[/]")
                    await stream_text(result.data.response)
                    console.print()  # Add a newline after response

if __name__ == '__main__':
    asyncio.run(main()) 