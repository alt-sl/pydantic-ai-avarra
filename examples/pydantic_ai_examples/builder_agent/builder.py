"""Builder Agent Implementation

This module implements a Builder Agent that can design and configure other agents based on
natural language descriptions. The Builder Agent can:
1. Understand requirements for a new agent
2. Design appropriate system prompts
3. Configure the agent with appropriate settings
4. Hand off conversation to the created agent
"""

from dataclasses import dataclass
from typing import Optional, Literal, Union, List

import logfire
from pydantic import BaseModel, Field
from rich.prompt import Prompt

from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import Usage, UsageLimits

# Configure logfire - 'if-token-present' means nothing will be sent if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

# Models to represent agent configurations and state

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
    'claude-3-5-sonnet-latest',  # Using Claude Sonnet for its superior capabilities
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
    """Create a new agent with the given configuration.
    
    This tool validates the configuration and creates a new agent instance.
    """
    # Store the configuration for future reference
    ctx.deps.current_config = config
    
    # Create the new agent
    new_agent = Agent[None, str](
        config.model_name,
        system_prompt=config.system_prompt,
        name=config.name,
    )
    
    # Store the created agent
    ctx.deps.created_agent = new_agent
    
    return config

@builder_agent.tool
async def handoff_to_agent(ctx: RunContext[BuilderDeps], query: str) -> AgentResponse:
    """Hand off the conversation to the created agent.
    
    This tool checks if an agent exists and then runs it with the given query.
    """
    if not ctx.deps.created_agent:
        raise ModelRetry("No agent has been created yet. Please create an agent first.")
    
    # Use non-streaming response since Anthropic doesn't support streaming yet
    result = await ctx.deps.created_agent.run(query, message_history=ctx.deps.message_history)
    
    # Store the messages for context continuity
    ctx.deps.message_history.extend(result.all_messages())
    
    return AgentResponse(response=result.data)

async def main():
    """Example usage of the Builder Agent."""
    # Initialize dependencies
    deps = BuilderDeps()
    
    while True:
        # Get the user's request
        request = Prompt.ask(
            'What kind of agent would you like to create? (or "quit" to exit)',
            default="quit"
        )
        
        if request.lower() == "quit":
            break

        # Create the agent based on the description
        result = await builder_agent.run(request, deps=deps)
        
        if isinstance(result.data, AgentConfig):
            print(f"\nCreated agent configuration:")
            print(f"Name: {result.data.name}")
            print(f"Description: {result.data.description}")
            print(f"Model: {result.data.model_name}")
            print(f"System Prompt: {result.data.system_prompt}")
            print("\nConfiguration complete!")
        
            # Now we can interact with the created agent
            while True:
                query = Prompt.ask(
                    '\nWhat would you like to ask the agent? (or "back" to create a new agent)',
                    default="back"
                )
                
                if query.lower() == "back":
                    break
                    
                # Get the agent's response
                result = await builder_agent.run(query, deps=deps)
                if isinstance(result.data, AgentResponse):
                    print(f"\nAgent response:\n{result.data.response}")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main()) 