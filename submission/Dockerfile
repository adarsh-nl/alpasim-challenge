FROM python:3.12-slim

RUN pip install --no-cache-dir \
    grpcio \
    grpcio-tools \
    "protobuf>=4.0.0,<5.0.0" \
    "setuptools<82" \
    "dataclasses-json>=0.6.7" \
    numpy

# vendored alpasim_grpc proto stubs
COPY grpc /tmp/alpasim_grpc
RUN pip install --no-cache-dir /tmp/alpasim_grpc && rm -rf /tmp/alpasim_grpc

RUN groupadd --system alpasim && \
    useradd --system --gid alpasim --home-dir /app --shell /usr/sbin/nologin alpasim

WORKDIR /app
COPY route_follower.py /app/driver.py
RUN chown -R alpasim:alpasim /app

ENV ALPASIM_DRIVER_HOST=0.0.0.0
ENV ALPASIM_DRIVER_PORT=6789
ENV RF_V_MAX=20 RF_A_LON=6.0 RF_A_LAT=7.0
EXPOSE 6789

USER alpasim
CMD ["python", "/app/driver.py"]
