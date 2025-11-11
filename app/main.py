import joblib
import pandas as pd
from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import warnings
import logging
import time
import json

# OpenTelemetry imports for tracing
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# Suppress specific scikit-learn warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)

app = FastAPI(
    title="Iris Species Predictor API",
    description="An API to predict the species of an Iris flower based on its measurements.",
    version="1.0.0"
)

# Setup OpenTelemetry Tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Setup structured logging
logger = logging.getLogger("iris-predictor-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()

formatter = logging.Formatter(json.dumps({
    "severity": "%(levelname)s",
    "message": "%(message)s",
    "timestamp": "%(asctime)s"
}))
handler.setFormatter(formatter)
logger.addHandler(handler)


# Define the input data model using Pydantic for validation
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        }

# Global variable to hold the loaded model
model = None

# Simulated flags for application state (e.g., for readiness/liveness probes)
app_state = {"is_ready": False, "is_alive": True}

@app.on_event("startup")
async def startup_event():
    """
    Event handler that runs when the FastAPI application starts up.
    It loads the machine learning model and updates the application's readiness state.
    """
    global model
    logger.info(json.dumps({"event": "startup", "message": "Starting model loading..."}))
    try:
        # Simulate work if model loading is fast, otherwise it's just the loading time
        # time.sleep(2)
        model = joblib.load("models/model.joblib")
        app_state["is_ready"] = True
        logger.info(json.dumps({"event": "startup", "message": "Model loaded successfully."}))
    except Exception as e:
        logger.exception(json.dumps({
            "event": "startup_error",
            "message": "Failed to load model. Application will not be ready.",
            "error": str(e)
        }))
        app_state["is_alive"] = False  # Mark as not alive if model loading fails critically
        app_state["is_ready"] = False

@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    """
    Liveness probe endpoint to check if the application is running.
    Returns 200 OK if the application is alive, 500 Internal Server Error otherwise.
    """
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    """
    Readiness probe endpoint to check if the application is ready to serve requests.
    Returns 200 OK if the model is loaded, 503 Service Unavailable otherwise.
    """
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    HTTP middleware to add a custom header with the request processing time.
    """
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler to catch all unhandled exceptions, log them,
    and return a standardized JSON error response with a trace ID.
    """
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

@app.post("/predict")
async def predict(iris: IrisFeatures, request: Request):
    """Predict the Iris species from input features."""
    if not app_state["is_ready"]:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded yet.")

    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            input_data = iris.dict()
            data = pd.DataFrame([input_data])
            prediction = model.predict(data)[0]
            latency = round((time.time() - start_time) * 1000, 2)

            # Log successful prediction with structured data
            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": input_data,
                "result": {"species": prediction},
                "latency_ms": latency,
                "status": "success"
            }))
            return {"species": prediction}

        except Exception as e:
            # Set span status to error and log the exception
            span.set_status(trace.Status(trace.StatusCode.ERROR, description=str(e)))
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e),
                "input": iris.dict(),
                "status": "failure"
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")
