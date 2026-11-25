"""Minimal serialized reconciliation, with explicit API consistency assumptions."""
import random
import threading

class PodController:
    def __init__(self,api,min_pods,max_pods):
        if not 0<=min_pods<=max_pods:raise ValueError('invalid range')
        self.api,self.min_pods,self.max_pods=api,min_pods,max_pods
        self.lock=threading.Lock()

    def reconcile(self):
        # Single writer; list is authoritative and counts created-but-starting pods.
        with self.lock:
            pods=self.api.list_pods()
            while len(pods)<self.min_pods:
                self.api.create_pod()
                pods=self.api.list_pods()
            while len(pods)>self.max_pods:
                self.api.delete_pod(pods[-1].name)
                pods=self.api.list_pods()
            return len(pods)

    def run(self,stop,interval=1.,max_backoff=30.,on_error=None):
        backoff=interval
        while not stop.is_set():
            try:
                self.reconcile();backoff=interval;delay=interval
            except Exception as exc:
                if on_error:on_error(exc)
                delay=backoff*random.uniform(.8,1.2)
                backoff=min(max_backoff,backoff*2)
            stop.wait(delay)
