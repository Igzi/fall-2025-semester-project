# Basic image based on ubuntu 22.04
FROM docker.io/library/ubuntu:22.04

#####################################
# RCP CaaS requirement (Storage)
#####################################
# Create your user inside the container.
# This block is needed to correctly map
# your EPFL user id inside the container.
# Without this mapping, you are not able
# to access files from the external storage.
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

# Copy your code inside the container
RUN mkdir -p /home/${LDAP_USERNAME}
COPY ./ /home/${LDAP_USERNAME}

# Set your user as owner of the new copied files
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}

# Install packages
RUN apt update
RUN apt update && apt install -y python3.10 python3.10-distutils python3-pip

# Set the working directory of the container
WORKDIR /home/${LDAP_USERNAME}

# Install Python dependencies
RUN pip install -e peft/
RUN pip install -r requirements.txt

# Try to detect NVIDIA GPU and install faiss-gpu if present
RUN if lspci | grep -i nvidia; then \
		pip uninstall -y faiss-cpu; \
		pip install faiss-gpu; \
	fi

USER ${LDAP_USERNAME}