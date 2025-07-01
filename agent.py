# --- Load environment variables (API keys and secrets) ---
from dotenv import load_dotenv        


# --- Import all core LiveKit Agent framework and plugin modules needed for AI, speech, and audio ---
from livekit import agents             
from livekit.agents import AgentSession, Agent, RoomInputOptions   
from livekit.plugins import (
    openai,           # OpenAI for language (LLM)
    cartesia,         # Cartesia for text-to-speech (TTS)
    deepgram,         # Deepgram for speech-to-text (STT)
    noise_cancellation,  # Noise cancellation
    silero,           # Voice activity detection (VAD)
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel  

# --- Initialize environment by loading variables from the .env file ---
load_dotenv()    

# --- Define your custom AI assistant by extending the Agent class ---
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")

# --- Set up and run the main agent session pipeline (STT → LLM → TTS, etc.) ---
async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(model="sonic-2", voice="791d5162-d5eb-40f0-8189-f19db44611d8"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(), 
        ),
    )

    await ctx.connect()      
    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )

# --- Entry point: Run the agent if this script is run directly ---
if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
