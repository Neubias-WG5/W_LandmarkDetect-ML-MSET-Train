FROM neubiaswg5/neubias-base

RUN pip install scikit-learn imageio scipy
RUN pip install https://github.com/Cytomine-ULiege/LandmarkTools/archive/master.zip

ADD descriptor.json /app/descriptor.json
ADD run.py /app/run.py
ADD download.py /app/download.py

ENTRYPOINT ["python", "/app/run.py"]