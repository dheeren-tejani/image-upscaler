# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from app.apis.routes import chat as chat_router # If you have this
# from app.services import local_embedding_service # If you have this
from app.apis.routes import super_resolution as sr_router
from app.core.config import settings
# Import the service module to trigger its initial loading attempts
from app.services import super_resolution_service 

app = FastAPI(title=settings.APP_NAME)

# CORS Middleware (ensure this is correctly configured)
origins = [
    "http://localhost:5173", # Default Vite React dev port
    "http://localhost:3000", # Common Create React App port
    "http://localhost:8080"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print(f"--- {settings.APP_NAME} starting up ---")
    # The import of super_resolution_service should have triggered its module-level
    # load_sr_model() call. We can check its status here.
    loaded_model = super_resolution_service.get_sr_model_instance() # This will try to load if still None
    if loaded_model:
        print("MAIN_APP: SR model confirmed loaded on startup.")
    else:
        print("MAIN_APP WARNING: SR model FAILED to load on startup. Upscaling endpoint will likely fail or try to reload.")
    
    # If you have other services to initialize (like local_embedding_service for the other project):
    # if hasattr(local_embedding_service, 'model') and local_embedding_service.model is None:
    #     print("MAIN_APP: Local embedding model not loaded. Initializing...")
    #     local_embedding_service.initialize_local_embedding_model()
    # else:
    #     print("MAIN_APP: Local embedding model status checked.")
    print("--- Application startup sequence complete ---")


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"Welcome to {settings.APP_NAME}. FastAPI server is running."}

# Include SR router
app.include_router(sr_router.router, prefix="/api/v1/sr", tags=["Super Resolution"])
# app.include_router(chat_router.router, prefix="/api/v1", tags=["Chat & Data"]) # If you have it