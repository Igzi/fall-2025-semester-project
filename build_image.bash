docker build --platform linux/amd64 . --tag registry.rcp.epfl.ch/igor-pavlovic-semester-project-fall-2025/lora-retriever:0.2 \
    --build-arg LDAP_GROUPNAME=rcp-runai-sacs \
    --build-arg LDAP_GID=30217 \
    --build-arg LDAP_USERNAME=pavlovic \
    --build-arg LDAP_UID=287589