from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware


from util import load_model, pre_process, post_process 

model = load_model()

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    max_age=3600,
)

@app.post("/upload")
async def create_upload_files(request: Request):
    
    data = await request.json()
    image = data.get('image')
    
    # Return preprocessed input batch and loaded image
    image = pre_process(image)

    # Run the model and postpocess the output
    prediction = model.predict(image)

    # Post process and stitch together the two images to return them
    res = post_process(prediction)
    return res


@app.get("/")
async def main():
 
    return {"message" : "success"}
