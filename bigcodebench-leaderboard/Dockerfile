FROM bigcodebench/bigcodebench-gradio:latest
COPY . /app
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]