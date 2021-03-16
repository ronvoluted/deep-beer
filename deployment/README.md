# Running API Locally

```bash
cd deep-beer/deployment

pipenv run uvicorn main:app --reload
```

Visit [localhost:8000](localhost:8000)

# Building Docker Image and Running Container
```bash
cd deep-beer/deployment

docker build -t deep-beer -f  deployment/Dockerfile .

docker run -itp 8000:80 --name fastapi-deploy deep-beer
```

Visit [localhost:8000](localhost:8000)

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
