"""W545 parking OOD with composition and a lock-protected in-memory store."""

from dataclasses import dataclass
from enum import IntEnum
import math
import threading


class VehicleSize(IntEnum):
    MOTORCYCLE = 0
    CAR = 1
    TRUCK = 2


@dataclass(frozen=True)
class Vehicle:
    plate: str
    size: VehicleSize
    required_features: frozenset = frozenset()


@dataclass(frozen=True)
class Spot:
    identifier: str
    capacity: VehicleSize
    features: frozenset = frozenset()

    def accepts(self, vehicle):
        return (
            self.capacity >= vehicle.size and vehicle.required_features <= self.features
        )


@dataclass(frozen=True)
class Ticket:
    identifier: int
    spot_id: str
    vehicle: Vehicle
    start_seconds: float


class HourlyPricing:
    def __init__(self, minor_units_per_hour):
        self.rate = minor_units_per_hour

    def cost(self, ticket, end_seconds):
        if end_seconds < ticket.start_seconds:
            raise ValueError("Time moved backwards")
        return (
            max(1, math.ceil((end_seconds - ticket.start_seconds) / 3600)) * self.rate
        )


class ParkingLot:
    def __init__(self, spots, pricing):
        self.spots = {spot.identifier: spot for spot in spots}
        if len(self.spots) != len(spots):
            raise ValueError("Duplicate spot ID")
        self.pricing = pricing
        self.active = {}
        self.receipts = {}
        self.next_id = 1
        self.lock = threading.Lock()

    def park(self, vehicle, now_seconds):
        with self.lock:
            if any(t.vehicle.plate == vehicle.plate for t in self.active.values()):
                raise ValueError("Vehicle already parked")
            occupied = {t.spot_id for t in self.active.values()}
            choices = [
                s
                for s in self.spots.values()
                if s.identifier not in occupied and s.accepts(vehicle)
            ]
            if not choices:
                raise ValueError("No compatible free spot")
            spot = min(choices, key=lambda s: (s.capacity, s.identifier))
            ticket = Ticket(self.next_id, spot.identifier, vehicle, now_seconds)
            self.next_id += 1
            self.active[ticket.identifier] = ticket
            return ticket

    def leave(self, ticket_id, now_seconds):
        with self.lock:
            if ticket_id in self.receipts:
                return self.receipts[ticket_id]
            ticket = self.active[ticket_id]
            amount = self.pricing.cost(ticket, now_seconds)
            receipt = dict(
                ticket_id=ticket_id, amount_minor=amount, spot_id=ticket.spot_id
            )
            self.receipts[ticket_id] = receipt
            del self.active[ticket_id]
            return receipt


if __name__ == "__main__":
    lot = ParkingLot(
        [Spot("C", VehicleSize.CAR), Spot("T", VehicleSize.TRUCK)], HourlyPricing(200)
    )
    car = lot.park(Vehicle("A", VehicleSize.CAR), 0)
    assert car.spot_id == "C"
    truck = lot.park(Vehicle("B", VehicleSize.TRUCK), 0)
    assert truck.spot_id == "T"
    try:
        lot.park(Vehicle("C", VehicleSize.CAR), 0)
        raise AssertionError("Expected full")
    except ValueError:
        pass
    first = lot.leave(car.identifier, 3601)
    assert first["amount_minor"] == 400
    assert lot.leave(car.identifier, 9999) == first
    ev = ParkingLot(
        [Spot("EV", VehicleSize.CAR, frozenset({"charger"}))], HourlyPricing(300)
    )
    assert (
        ev.park(Vehicle("E", VehicleSize.CAR, frozenset({"charger"})), 0).spot_id
        == "EV"
    )
    print("W545 fitting, full lot, pricing, retry and EV composition checks passed")
