FROM public.ecr.aws/lambda/python:3.7
COPY requirements.txt ./
COPY modnet_photographic_portrait_matting.ckpt./
ADD output output
RUN chmod -R 777 output
ADD src src
ADD test test
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY inference.py
CMD ["inference.handler"}
