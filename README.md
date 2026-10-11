pi-modal
========

Auto-discover available models from [Modal](https://modal.com/) shared
inference endpoints


installation
------------

```
    % pi install ~/src/pi-modal
```


usage
-----

```
    % export MODAL_PROXY_TOKEN='wk-<id>.ws-<secret>'
    % export MODAL_BASE_URL='https://inference.us-west.modal.direct/v1'    # optional
    % pi

    /model
```

See https://modal.com/docs/guide/endpoints#proxy-tokens for more
details on proxy tokens.


compatibility
-------------

This is a personal project.  You're free to use it, but no
compatibility or stability between changes is guaranteed.
