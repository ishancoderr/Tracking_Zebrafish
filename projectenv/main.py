from fastapi import FastAPI, Depends
from fastapi import FastAPI, HTTPException
from api.preprocessing.videoProcessor import Preprocessing
from api.preprocessing.preprocessingSteps import PreprocessingSteps
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI(
        title="Fish Tracking API", 
        description="Tracking Zebrafish in Laboratory Conditions",
        version="1.0.0",
        swagger_ui_parameters={"defaultModelsExpandDepth": -1}
    )

@app.get("/")
async def root():
    return {"message": "Welcome to the Fish Tracking API!"}

@app.post("/get_detectFish_v1")
async def detectFishByAlgorithems_v1(directory_path: str, output_path: str):
    try:
        preprocessing = Preprocessing(directory_path=directory_path, output_path=output_path)
        
        video_paths = await preprocessing.getVideoPaths()
        
        filtered_video_paths = await preprocessing.addFilters(video_paths, filter_name='no filter')

        processed_video_paths = []
        background_sub_video_paths = []
        
        for video_path in filtered_video_paths:
            result = await preprocessing.backgroundSubtraction(video_path)
            
            processed_video_paths.append(result['processed_video'])
            background_sub_video_paths.append(result['bg_subtracted_video'])

        return {
            "processed_videos": processed_video_paths,
            "background_sub_videos": background_sub_video_paths
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/get_detectFish_v2")
async def detectFishByAlgorithems_v2(directory_path: str, output_path: str):
    try:
        preprocessing = PreprocessingSteps(directory_path=directory_path, output_path=output_path)
        print(preprocessing)
        video_paths = await preprocessing.getVideoPaths()
        
        filtered_video_paths = await preprocessing.addFilters(video_paths, filter_name='no filter')

        processed_video_paths = []
        background_sub_video_paths = []
        heatmap_paths = []
        trajectory_paths = []
        for video_path in filtered_video_paths:
           result = await preprocessing.getVideoFrames(video_path)
            
           processed_video_paths.append(result['processed_video'])
           background_sub_video_paths.append(result['bg_subtracted_video'])
           heatmap_path = result['heatmap']
           trajectory_path = result['trajectory_plot']
            
           heatmap_paths.append(heatmap_path)
           trajectory_paths.append(trajectory_path)

        return {
            "processed_videos": processed_video_paths,
            "background_sub_videos": background_sub_video_paths,
            "heatmaps": heatmap_paths,
            "trajectory_plots": trajectory_paths
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")    
 

