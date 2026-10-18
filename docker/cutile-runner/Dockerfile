FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    RUNNER_ALLOW_RUNASROOT=0 \
    HOME=/home/runner

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git jq \
      libicu74 libkrb5-3 libssl3t64 zlib1g libstdc++6 libgcc-s1 \
      libcurl4t64 libunwind8 liblttng-ust1t64 \
      python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash runner

RUN set -eux; \
    version="$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest | jq -r .tag_name | sed 's/^v//')"; \
    test -n "$version"; \
    curl -fsSL "https://github.com/actions/runner/releases/download/v${version}/actions-runner-linux-x64-${version}.tar.gz" -o /tmp/runner.tar.gz; \
    mkdir -p /home/runner/actions-runner; \
    tar -xzf /tmp/runner.tar.gz -C /home/runner/actions-runner; \
    rm /tmp/runner.tar.gz; \
    chown -R runner:runner /home/runner

COPY --chown=root:root entrypoint.sh /usr/local/bin/cutile-runner-entrypoint
RUN chmod 0555 /usr/local/bin/cutile-runner-entrypoint

USER runner
WORKDIR /home/runner/actions-runner
ENTRYPOINT ["/usr/local/bin/cutile-runner-entrypoint"]
