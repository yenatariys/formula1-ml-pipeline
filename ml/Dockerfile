FROM python:3.10
WORKDIR /app
RUN pip install pandas scikit-learn psycopg2-binary sqlalchemy
COPY train.py .
CMD ["python", "train.py"]