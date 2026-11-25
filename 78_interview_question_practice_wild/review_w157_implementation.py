"""W157 executable answer; demonstration fixtures are explicitly labeled."""

from dataclasses import dataclass
import time


@dataclass(frozen=True)
class Instance:
    id: str
    provider: str
    name: str
    gpu_type: str
    count: int
    region: str
    state: str


class ProviderError(Exception):
    pass


class Facade:
    # Adapter contract: list/get/create/stop and optional wait_operation.
    # create and stop return a stable instance ID and optional operation ID.
    def __init__(self, providers, clock=time.monotonic, sleep=time.sleep):
        self.providers, self.clock, self.sleep = providers, clock, sleep

    def list(self, provider=None):
        names = [provider] if provider else sorted(self.providers)
        instances, errors = [], {}
        for name in names:
            try:
                instances.extend(self.providers[name].list())
            except ProviderError as exc:
                errors[name] = str(exc)
        return instances, errors

    def get(self, provider, instance_id):
        return self.providers[provider].get(instance_id)

    def _finish(self, adapter, instance_id, operation, target, timeout):
        deadline = self.clock() + timeout
        while True:
            # The concrete adapter must apply a per-RPC deadline too.
            if operation:
                adapter.check_operation(operation)  # raises on operation failure
            item = adapter.get(instance_id)
            if item.state == target:
                return item
            if item.state == "FAILED":
                raise ProviderError(f"instance {instance_id} failed")
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise TimeoutError(f"timed out; reconcile instance {instance_id}")
            self.sleep(min(0.25, remaining))

    def create(self, provider, gpu, count, name="", region="", timeout=120):
        if count < 1:
            raise ValueError("GPU count must be positive")
        adapter = self.providers[provider]
        instance_id, operation = adapter.create(gpu, count, name, region)
        return self._finish(adapter, instance_id, operation, "RUNNING", timeout)

    def stop(self, provider, instance_id, timeout=120):
        adapter = self.providers[provider]
        _, operation = adapter.stop(instance_id)
        return self._finish(adapter, instance_id, operation, "STOPPED", timeout)


# DEMONSTRATION TRANSPORT ONLY: no external endpoints or provider SDKs.
class DemoAdapter:
    def __init__(
        self,
        name,
        asynchronous=False,
        unavailable=False,
        fail_operation=False,
        stuck=False,
    ):
        self.name, self.asynchronous = name, asynchronous
        self.unavailable, self.fail_operation = unavailable, fail_operation
        self.stuck, self.instances, self.pending = stuck, {}, {}
        self.counter = 0
        self.last_scope = None

    def _healthy(self):
        if self.unavailable:
            raise ProviderError(f"{self.name}: provider unavailable")

    def list(self):
        self._healthy()
        return list(self.instances.values())

    def get(self, instance_id):
        self._healthy()
        if instance_id not in self.instances:
            raise ProviderError(f"{self.name}: instance not found: {instance_id}")
        return self.instances[instance_id]

    def create(self, gpu, count, name, region):
        self._healthy()
        if count > 8:
            raise ProviderError(f"{self.name}: quota exceeded")
        self.counter += 1
        ident = f"{self.name}-{self.counter}"
        self.instances[ident] = Instance(
            ident,
            self.name,
            name,
            gpu,
            count,
            region,
            "PENDING" if self.asynchronous else "RUNNING",
        )
        operation = f"create-{ident}" if self.asynchronous else None
        if operation:
            self.pending[operation] = (ident, "RUNNING", 0)
        return ident, operation

    def stop(self, instance_id):
        from dataclasses import replace

        item = self.get(instance_id)
        if item.state == "STOPPED":
            return instance_id, None
        self.instances[instance_id] = replace(
            item, state="STOPPING" if self.asynchronous else "STOPPED"
        )
        operation = f"stop-{instance_id}" if self.asynchronous else None
        if operation:
            self.pending[operation] = (instance_id, "STOPPED", 0)
        return instance_id, operation

    def check_operation(self, operation):
        from dataclasses import replace

        self._healthy()
        if self.fail_operation:
            raise ProviderError(f"{self.name}: operation failed")
        ident, target, polls = self.pending[operation]
        polls += 1
        self.pending[operation] = ident, target, polls
        if polls >= 2 and not self.stuck:
            self.instances[ident] = replace(self.instances[ident], state=target)


def main(argv=None):
    import argparse
    import json
    import sys
    from dataclasses import asdict

    names = ["crusoe", "lambda", "nebius"]
    parser = argparse.ArgumentParser(
        description="DEMO vm CLI; adapters use in-memory fixtures"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--provider", choices=names)
    creation = commands.add_parser("create")
    creation.add_argument("--provider", choices=names, required=True)
    creation.add_argument("--gpu", required=True)
    creation.add_argument("--count", type=int, required=True)
    creation.add_argument("--name", default="")
    creation.add_argument("--region", default="")
    for command in ("get", "stop"):
        sub = commands.add_parser(command)
        sub.add_argument("--provider", choices=names, required=True)
        sub.add_argument("instance_id")
    args = parser.parse_args(argv)
    adapters = {n: DemoAdapter(n, asynchronous=n != "lambda") for n in names}
    # Fixed demo inventory makes get/stop independently executable.
    for n, adapter in adapters.items():
        adapter.instances[f"{n}-demo"] = Instance(
            f"{n}-demo", n, "demo", "DEMO-GPU", 1, "demo-region", "RUNNING"
        )
    facade = Facade(adapters)
    try:
        if args.command == "list":
            items, errors = facade.list(args.provider)
            print(json.dumps([asdict(item) for item in items]))
            for name, message in errors.items():
                print(f"{name}: {message}", file=sys.stderr)
            return 2 if errors else 0
        if args.command == "create":
            item = facade.create(
                args.provider, args.gpu, args.count, args.name, args.region
            )
        elif args.command == "get":
            item = facade.get(args.provider, args.instance_id)
        else:
            item = facade.stop(args.provider, args.instance_id)
        print(json.dumps(asdict(item)))
        return 0
    except (ProviderError, ValueError, TimeoutError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
