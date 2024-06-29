import cProfile
import pstats
import uvicorn
from main import app  # Ensure to import your FastAPI app

def profile_app():
    profiler = cProfile.Profile()
    profiler.enable()

    # Start the server in a separate thread
    uvicorn.run(app, host="127.0.0.1", port=8000)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.dump_stats('profile.stats')
    stats.print_stats()

if __name__ == "__main__":
    profile_app()
