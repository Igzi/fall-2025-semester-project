docker build --platform linux/amd64 . --tag registry.rcp.epfl.ch/sacs-pavlovic/my-toolbox:v0.1 \
    --build-arg LDAP_GROUPNAME=rcp-runai-sacs \
    --build-arg LDAP_GID=30217 \
    --build-arg LDAP_USERNAME=pavlovic \
    --build-arg LDAP_UID=287589