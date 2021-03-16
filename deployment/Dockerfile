FROM python:3.9.1

WORKDIR /home/beer

COPY . .

RUN pip install category-encoders==2.2.2
RUN pip install fastapi==0.63.0
RUN pip install numpy==1.20.1
RUN pip install pandas==1.2.3
RUN pip install scikit-learn==0.24.1
RUN pip install torch==1.8.0
RUN pip install uvicorn==0.13.4

CMD ["uvicorn", "main:app"]
