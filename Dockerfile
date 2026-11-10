# CalibForge: one image that runs the ROS 2 nodes, calibration, ML and tests.
#
# Lyrical runs on Ubuntu 26.04, so Python here is 3.14. rclpy and cv_bridge are
# built for that interpreter, which means everything below must install into it.
FROM ros:lyrical-ros-base-resolute

# LightGlue is not on PyPI, so it is pinned to a commit.
ARG LIGHTGLUE_COMMIT=5a650c444718caffc2669dc0094cde908f42dd73

# System and ROS 2 packages:
#   cv-bridge, tf2-ros-py, rosbag2 + MCAP   image conversion, TF and bag files
#   foxglove-bridge                         lets Foxglove connect to live topics
#   ros-testing                             node-level tests
#   libgl1, libgomp1                        runtime libraries for Open3D
#   git, make                               DVC, pre-commit and `make demo`
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libgomp1 \
        make \
        python3-pip \
        ros-${ROS_DISTRO}-cv-bridge \
        ros-${ROS_DISTRO}-foxglove-bridge \
        ros-${ROS_DISTRO}-ros-testing \
        ros-${ROS_DISTRO}-rosbag2 \
        ros-${ROS_DISTRO}-rosbag2-storage-mcap \
        ros-${ROS_DISTRO}-tf2-ros-py \
    && rm -rf /var/lib/apt/lists/*

# Install into the system Python rather than a venv so these packages sit next
# to rclpy: ROS node scripts run under /usr/bin/python3 and would not see a venv.
# --ignore-installed is needed because pip cannot uninstall apt-owned packages.
# It leaves them in place and installs to /usr/local, which takes priority.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages --ignore-installed \
        -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# --no-deps: LightGlue requires opencv-python, which would overwrite the
# headless cv2. Its real dependencies are already installed above.
RUN pip install --no-cache-dir --break-system-packages --no-deps \
    "lightglue @ https://github.com/cvg/LightGlue/archive/${LIGHTGLUE_COMMIT}.tar.gz"

WORKDIR /workspace
