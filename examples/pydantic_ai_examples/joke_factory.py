from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits

# Create the agents
joke_selection_agent = Agent(
    'openai:gpt-4o',  # Changed from gpt-4o to gpt-4 since that's the actual model name
    system_prompt=(
        'Use the `joke_factory` to generate some jokes, then choose the best. '
        'You must return just a single joke.'
    ),
)

joke_generation_agent = Agent('openai:gpt-4o', result_type=list[str])  # Updated to available Gemini model

@joke_selection_agent.tool
async def joke_factory(ctx: RunContext[None], count: int) -> list[str]:
    r = await joke_generation_agent.run(
        f'Please generate {count} jokes.',
        usage=ctx.usage,
    )
    return r.data

async def main():
    result = await joke_selection_agent.run(
        'Tell me a joke.',
        usage_limits=UsageLimits(request_limit=5, total_tokens_limit=300),
    )
    print(result.data)
    print(result.usage())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())