"""W228 cleaning and W246 scheduling demo; standard library."""
# BEGIN W228
import csv
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation

PAYMENT_ALIASES = {'card':'card', 'credit card':'card', 'credit_card':'card',
                   'credit crad':'card', 'cash':'cash', 'paypal':'paypal', 'pay pal':'paypal'}

def clean_transactions(path, id_width=None):
    clean, rejects = [], []
    with open(path, newline='', encoding='utf-8') as f:
        for line, row in enumerate(csv.DictReader(f), 2):
            try:
                date = None
                for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%b %d, %Y'):
                    try: date = datetime.strptime(row['date'].strip(), fmt).date(); break
                    except ValueError: pass
                if date is None: raise ValueError('unsupported date')
                # Explicit USD-only grammar avoids silently accepting mixed currencies.
                raw = re.sub(r'^(?:USD\s*|\$\s*)', '', row['amount'].strip(), flags=re.I)
                if not re.fullmatch(r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?', raw):
                    raise ValueError('invalid USD amount')
                amount = Decimal(raw.replace(',', ''))
                if not amount.is_finite(): raise ValueError('nonfinite amount')
                customer = row['customer_id'].strip()
                if not customer: raise ValueError('missing customer_id')
                if id_width is not None:
                    if not customer.isdigit() or len(customer) > id_width:
                        raise ValueError('customer ID violates declared fixed width')
                    customer = customer.zfill(id_width)
                payment = PAYMENT_ALIASES.get(row['payment_method'].strip().lower())
                if payment is None: raise ValueError('unknown payment method')
                clean.append(dict(date=date.isoformat(), amount=str(amount.quantize(Decimal('.01'))),
                                  customer_id=customer, payment_method=payment, notes=row.get('notes','')))
            except (ValueError, KeyError, InvalidOperation) as exc:
                rejects.append(dict(line=line,raw=row,reason=str(exc)))
    return clean, rejects
# END W228

# BEGIN W246

def earliest_meeting(slots_a, slots_b, duration):
    """DEMO contract: half-open integer intervals, positive integer duration."""
    if not isinstance(duration, int) or isinstance(duration, bool) or duration <= 0:
        raise ValueError('duration must be a positive integer')
    def validate(slots):
        if len(slots) > 10000: raise ValueError('too many slots')
        for slot in slots:
            if len(slot) != 2 or any(not isinstance(x,int) or isinstance(x,bool) for x in slot) or slot[0] >= slot[1]:
                raise ValueError('invalid interval')
        return sorted(slots)
    a, b = validate(slots_a), validate(slots_b)
    i = j = 0
    while i < len(a) and j < len(b):
        start, end = max(a[i][0],b[j][0]), min(a[i][1],b[j][1])
        if end - start >= duration: return [start,start+duration]
        if a[i][1] <= b[j][1]: i += 1
        else: j += 1
    return None

class SchedulingRepository:
    """Injected boundary; booking persistence would replace this read-only demo."""
    def availability(self, request):
        return request['slots_a'], request['slots_b']

class SchedulingService:
    def __init__(self, repository): self.repository = repository
    def find(self, request):
        a,b = self.repository.availability(request)
        return earliest_meeting(a,b,request['duration'])

class SchedulingHandler:
    def __init__(self, service): self.service = service
    def post(self, body):
        # Framework adapter would parse JSON, enforce body size/auth and add tracing.
        try:
            if not isinstance(body, dict) or set(body) != {'slots_a','slots_b','duration'}:
                raise ValueError('expected slots_a, slots_b, duration')
            return 200, {'meeting': self.service.find(body)}
        except (ValueError, TypeError, KeyError) as exc:
            return 422, {'error': 'invalid_request', 'message': str(exc)}
# END W246
