from dataclasses import dataclass
from typing import Optional, Literal, Union

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
    avatar: str = Field(
        description="An avatar for the agent"
    )

class AgentResponse(BaseModel):
    """Response from the created agent."""
    response: str = Field(description="The agent's response to the query")

@dataclass
class BuilderDeps:
    """Dependencies for the Builder Agent.
    
    This maintains state about the currently created agent and its configuration.
    """
    current_config: Optional[AgentConfig] = None
    created_agent: Optional[Agent[None, str]] = None

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
5. Once the agent is created, hand off the conversation completely to the new agent

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

Always validate that your configurations make sense for the user's needs.
After creating the agent, you will hand off the conversation completely to it."""
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
    
    # Use streaming for the created agent's response
    response_text = ""
    async with ctx.deps.created_agent.run_stream(query) as result:
        async for chunk in result.stream_text(delta=True):
            response_text += chunk
            print(chunk, end="", flush=True)  # Stream output directly
    
    return AgentResponse(response=response_text)

async def chat_with_agent(agent: Agent, user_message: str, message_history: Optional[list[ModelMessage]] = None) -> str:
    """Chat directly with an agent, bypassing the terminal interface."""
    async with agent.run_stream(
        user_message,
        message_history=message_history
    ) as result:
        response_text = ""
        async for chunk in result.stream_text(delta=True):
            response_text += chunk
        return response_text, result.new_messages()

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
        async with builder_agent.run_stream(request, deps=deps) as result:
            print("\nCreating agent configuration...")
            config = await result.get_data()
            if isinstance(config, AgentConfig):
                print(f"\nName: {config.name}")
                print(f"Description: {config.description}")
                print(f"Model: {config.model_name}")
                print(f"System Prompt: {config.system_prompt}")
                print("\nConfiguration complete! Now handing off to the created agent...")
            
            # Now we can interact directly with the created agent
            created_agent = deps.created_agent
            if not created_agent:
                print("Error: No agent was created")
                continue

            # Direct conversation loop with the created agent
            while True:
                query = Prompt.ask(
                    '\nWhat would you like to ask the agent? (or "back" to create a new agent)',
                    default="back"
                )
                
                if query.lower() == "back":
                    break
                    
                # Stream the created agent's response directly
                print("\nAgent response:")
                async with created_agent.run_stream(query) as result:
                    async for chunk in result.stream_text(delta=True):
                        print(chunk, end="", flush=True)  # Stream output directly
                print()  # Add newline after response

# Example usage:
async def direct_chat():
    # First create the agent using builder_agent
    deps = BuilderDeps()
    async with builder_agent.run_stream("Create a friendly chat assistant", deps=deps) as result:
        config = await result.get_data()
        
    created_agent = deps.created_agent
    if not created_agent:
        raise Exception("No agent was created")
    
    # Now we can chat directly with the created agent
    message_history = None
    
    # Example of direct chat interaction
    response, message_history = await chat_with_agent(
        created_agent, 
        "Hello! How are you?",
        message_history
    )
    
    # Continue conversation with history
    next_response, message_history = await chat_with_agent(
        created_agent,
        "Tell me more about yourself",
        message_history
    )

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())