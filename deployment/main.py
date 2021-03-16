from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from typing import Optional, List
from pydantic import BaseModel
import torch

from modules.pytorch import PytorchMultiClass
from modules.preprocess import process_input

description = """
# Project Objective

This is a custom neural network model trained to predict a type of beer based on brewery and user rating inputs. The model is hosted on Heroku to serve predictions, which can be fetched one at a time or as a list of items in the request.

 - brewery_name (str)
 - review_appearance (float)
 - review_aroma (float)
 - review_palate (float)
 - review_taste (float)

### Input Format (single)

```json
{
  "brewery_name": "Boston Beer Company (Samuel Adams)",
  "review_appearance": 3.8416471332705995,
  "review_aroma": 3.7356383055832003,
  "review_palate": 3.7437049311136588,
  "review_taste": 3.7928644856072644
}
```

### Output Format (single)

```json
{
  "brewery_name": "Boston Beer Company (Samuel Adams)",
  "review_appearance": 3.8416471332705995,
  "review_aroma": 3.7356383055832003,
  "review_palate": 3.7437049311136588,
  "review_taste": 3.7928644856072644,
  "beer_style": "Bock"
}
```

### Input Format (multiple)

```json
[
  {
    "brewery_name": "Zum LÃ¶wenbrÃ¤u",
    "review_appearance": 5,
    "review_aroma": 5,
    "review_palate": 5,
    "review_taste": 5
  },
  {
    "brewery_name": "Vecchio Birraio",
    "review_appearance": 1,
    "review_aroma": 1,
    "review_palate": 1,
    "review_taste": 1
  }
]
```

### Output Format (multiple)
```json
[
  {
    "brewery_name": "Zum LÃ¶wenbrÃ¤u",
    "review_appearance": 5,
    "review_aroma": 5,
    "review_palate": 5,
    "review_taste": 5,
    "beer_style": "Russian Imperial Stout"
  },
  {
    "brewery_name": "Vecchio Birraio",
    "review_appearance": 1,
    "review_aroma": 1,
    "review_palate": 1,
    "review_taste": 1,
    "beer_style": "Fruit / Vegetable Beer"
  }
]
```

### Heroku URL
[https://young-falls-22950.herokuapp.com](https://young-falls-22950.herokuapp.com)

### GitHub Repository
[https://github.com/ronvoluted/deep-beer](https://github.com/ronvoluted/deep-beer)

# Endpoints
"""

app = FastAPI(
    title="Ron Au | Assignment 2: Beer Review Project",
    description=description,
    version="1.0.2",
    docs_url="/"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=["*"],
    allow_headers=["*"]
)

NUM_FEATURE_COLUMNS = 107
NUM_BEER_STYLES = 104

model = PytorchMultiClass(NUM_FEATURE_COLUMNS, NUM_BEER_STYLES)
state = torch.load('./artefacts/deep_beer_state.pt', map_location=torch.device('cpu'))
model.load_state_dict(state['model_state_dict'])

class Review(BaseModel):
    brewery_name: Optional[str] = 'Boston Beer Company (Samuel Adams)'
    review_appearance: Optional[float] = 3.8416471332705995
    review_aroma: Optional[float] = 3.7356383055832003
    review_palate: Optional[float] = 3.7437049311136588
    review_taste: Optional[float] = 3.7928644856072644


@app.get("/", tags=['Root'])
def root():
    return {"Hello": "World"}

@app.get("/docs", tags=['Root'])
async def redirect_to_root():
    response = RedirectResponse(url="/")
    return response

@app.get('/health/', tags=['Meta'], status_code=200)
def health_check():
    return 'Enjoy responsibly!'

@app.get('/model/architecture/', tags=['Meta'])
def deep_beer_network():
    return model

@app.post('/beer/type/', tags=['Predictions'])
def predict_single(review: Review):
    return process_input(review, model)

@app.post('/beers/type/', tags=['Predictions'])
def predict_multiple(reviews: List[Review]):
    predictions = list()
    for review in reviews:
        predictions.append(process_input(review, model))
    return predictions
