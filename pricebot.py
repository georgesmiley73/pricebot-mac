#!/usr/bin/env python3
# pricebot_v6d.py — v5 UI + v6 algorithm improvements + v6a burn-in patch + v6b undo-fix + v6d production build

import json, os, sys, random
import numpy as np
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION & STATE
# ─────────────────────────────────────────────────────────────────────────────

HIST_FILE = Path.home() / ".pricebot_v6_state.json"

# ─── Controller parameters ────────────────────────────────────────────────────
Kp, Ki, Kd        = 1.2, 0.3, 0.06
DERIV_ALPHA       = 0.3
NEAR_SMOOTH, FAR_SMOOTH = 1.0, 0.5
DAMP_TABLE        = [(50,0.0),(60,0.2),(70,0.6),(80,0.8),(95,1.0),(600,1.0)]
BIG_ERR           = 1.0
HYSTERESIS_DAYS   = 2

# ─── Simulation parameters ────────────────────────────────────────────────────
WTP_LOW, WTP_HIGH = 70.0, 300.0
WTP_STD           = 20.0       # daily WTP volatility
N_BOOKERS         = 1_000      # potential bookers per day
TARGET_DEMAND     = 100.0      # percent
CLAMP_MIN, CLAMP_MAX = 40.0, 500.0
BURN_IN_DAYS      = 365        # one-year burn-in

# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def lookup(table, x):
    """Occupancy -> damping lookup."""
    for thresh, val in table:
        if x <= thresh:
            return val
    return table[-1][1]

def seasonal_mean(doy: int) -> float:
    """Piecewise constant 4-season WTP mean."""
    if   doy <  90: return (WTP_LOW + WTP_HIGH)/2
    elif doy < 180: return WTP_HIGH
    elif doy < 270: return (WTP_LOW + WTP_HIGH)/2
    else:           return WTP_LOW

# ─────────────────────────────────────────────────────────────────────────────
#  PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def load_state():
    if HIST_FILE.exists():
        try:
            return json.loads(HIST_FILE.read_text())
        except:
            pass
    # initial production-mode state
    return {
        # last computed values
        'near':100.0, 'far':100.0,
        'demand':TARGET_DEMAND, 'occ_near':TARGET_DEMAND, 'occ_far':TARGET_DEMAND,
        'min_near':CLAMP_MIN, 'max_near':CLAMP_MAX,
        'min_far':CLAMP_MIN, 'max_far':CLAMP_MAX,
        # smoothing/PID state (seven parameters to burn-in)
        'cum_rooms':0.0, 'cum_days':0.0,
        'avg_1':TARGET_DEMAND, 'avg_4':TARGET_DEMAND, 'avg_7':TARGET_DEMAND if False else TARGET_DEMAND, # placeholder
        'integral':0.0, 'prev_error':0.0, 'prev_derivative':0.0,
        'trend_buffer_above':0, 'trend_buffer_below':0,
        'consec_above':0, 'consec_below':0,
        # burn-in flag
        'burned_in':False,
        'seed_price':100.0
    }

def save_state(st):
    HIST_FILE.write_text(json.dumps(st))

# ─────────────────────────────────────────────────────────────────────────────
#  PER-ROOM STATE SUPPORT  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

ROOMS = ("Roxanne", "Persephone")

def hist_file(room: str) -> Path:
    """Return the JSON file path for one room."""
    return Path.home() / f".pricebot_{room}_state.json"

def load_state_for(room: str):
    """Load state for a room, falling back to factory defaults."""
    p = hist_file(room)
    global HIST_FILE
    old_hist = HIST_FILE
    HIST_FILE = p
    st = load_state()            # uses the room-specific path now
    HIST_FILE = old_hist
    return st

def save_state_for(room: str, st):
    hist_file(room).write_text(json.dumps(st))

# hold an independent state-dict for every room
state_by_room = {r: load_state_for(r) for r in ROOMS}

def activate_room(room: str):
    """Point the globals at the chosen room’s state & file."""
    global state, HIST_FILE
    state     = state_by_room[room]
    HIST_FILE = hist_file(room)

# make sure module-level code that follows has a valid `state`
activate_room(ROOMS[0])

# ─────────────────────────────────────────────────────────────────────────────
#  FIRST-USE BURN-IN PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def perform_burn_in(root):
    if state.get('burned_in'):
        return
    # Prompt user for season & WTP seed
    seasons = ['low','low→high','high','high→low']
    season = simpledialog.askstring("Initialize PriceBot",
        "First run: select current season:\n" +
        "\n".join(f"{i+1}. {s}" for i,s in enumerate(seasons)),
        parent=root)
    try:
        idx = int(season.strip()) - 1
        season = seasons[idx]
    except:
        messagebox.showerror("Error", "Invalid season selection. Defaulting to ‘low’.")
        season = 'low'
    wtp_str = simpledialog.askstring("Initialize PriceBot",
        f"Enter representative WTP for '{season}' season:", parent=root)
    try:
        wtp = float(wtp_str)
    except:
        messagebox.showerror("Error", "Invalid WTP. Defaulting to average.")
        wtp = seasonal_mean(0)

    # spin up all seven core-logic parameters over one full year at constant WTP
    price = state['seed_price']
    cum_rooms = 0.0
    cum_days  = 0.0
    sm4 = sm7 = TARGET_DEMAND
    integral = prev_error = prev_derivative = 0.0
    trend_buffer_above = trend_buffer_below = 0
    consec_above = consec_below = 0
    trend_dem = TARGET_DEMAND
    raw_dem = TARGET_DEMAND

    for _ in range(BURN_IN_DAYS):
        draw = max(0.0, random.gauss(wtp, WTP_STD))
        demand_pct = min(200.0, 100.0 * (draw / price))
        # accumulate for raw_dem
        rooms = demand_pct / 100.0
        cum_rooms += rooms
        cum_days  += 1.0
        raw_dem = (cum_rooms / cum_days) * 100.0
        # low-pass trend
        trend_dem = 0.01 * demand_pct + 0.99 * trend_dem
        hp = demand_pct - trend_dem
        # smoothing
        sm1 = hp
        sm4 = 0.75 * sm4 + 0.25 * hp
        sm7 = (6/7) * sm7 + (1/7) * hp
        sm  = 0.58 * sm1 + 0.29 * sm4 + 0.13 * sm7
        err = sm
        # hysteresis buffers
        if err >= BIG_ERR:
            trend_buffer_above += 1
            trend_buffer_below  = 0
        elif err <= -BIG_ERR:
            trend_buffer_below += 1
            trend_buffer_above  = 0
        else:
            trend_buffer_above = trend_buffer_below = 0
        # apply hysteresis
        if trend_buffer_above >= HYSTERESIS_DAYS:
            consec_above += 1
            consec_below  = 0
        elif trend_buffer_below >= HYSTERESIS_DAYS:
            consec_below += 1
            consec_above  = 0
        days_trend = max(consec_above, consec_below)
        factor = (0.0 if days_trend == 0 else
                  0.1 if days_trend == 1 else
                  0.2 if days_trend == 2 else 1.0)
        # PID
        if days_trend == 1 or prev_error * err < 0:
            integral = 0.0
        integral += err
        rawD = err - prev_error
        derivative = DERIV_ALPHA * rawD + (1 - DERIV_ALPHA) * prev_derivative
        delta = (Kp * err + Ki * integral + Kd * derivative) * factor
        # update price with damping
        raw_damp = lookup(DAMP_TABLE, demand_pct)

        damp = raw_damp if delta >= 0 else 1.0
        ns = NEAR_SMOOTH if days_trend < 3 else 1.0
        price *= (1 + (ns * damp * delta) / 100.0)
        price = min(CLAMP_MAX, max(CLAMP_MIN, price))
        # roll burn-in state
        prev_error, prev_derivative = err, derivative

    # persist all seven parameters back to state
    state.update({
        'near': price,
        'far': price,
        'cum_rooms': cum_rooms,
        'cum_days': cum_days,
        'avg_1': raw_dem,
        'avg_4': sm4,
        'avg_7': sm7,
        'integral': integral,
        'prev_error': prev_error,
        'prev_derivative': prev_derivative,
        'trend_buffer_above': trend_buffer_above,
        'trend_buffer_below': trend_buffer_below,
        'consec_above': consec_above,
        'consec_below': consec_below,
        'burned_in': True
    })
    save_state(state)
    messagebox.showinfo("Initialization Complete",
        f"Engine seeded at price €{price:.2f} after burn-in.")

# ─────────────────────────────────────────────────────────────────────────────
#  CORE ENGINE UPDATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_prices(pn, pf, dem, on, of, mn_n, mx_n, mn_f, mx_f):
    # 1) cumulative smoothing
    rooms = dem / 100.0
    state['cum_rooms'] += rooms
    state['cum_days']  += 1.0
    raw_dem = (state['cum_rooms'] / state['cum_days']) * 100.0
    # 1a) 1/4/7-day smoothing
    avg4 = 0.75 * state['avg_4'] + 0.25 * dem
    avg7 = (6/7) * state['avg_7'] + (1/7) * dem
    sm   = 0.58 * raw_dem + 0.29 * avg4 + 0.13 * avg7
    state.update(avg_1=raw_dem, avg_4=avg4, avg_7=avg7)
    # 2) error
    err = sm - TARGET_DEMAND
    # 3) hysteresis buffers
    if   err >= BIG_ERR:
        state['trend_buffer_above'] += 1; state['trend_buffer_below'] = 0
    elif err <= -BIG_ERR:
        state['trend_buffer_below'] += 1; state['trend_buffer_above'] = 0
    else:
        state['trend_buffer_above'] = state['trend_buffer_below'] = 0
    # 4) apply after HYSTERESIS_DAYS
    Ca, Cb = state['consec_above'], state['consec_below']
    if state['trend_buffer_above'] >= HYSTERESIS_DAYS:
        Ca += 1; Cb = 0
    elif state['trend_buffer_below'] >= HYSTERESIS_DAYS:
        Cb += 1; Ca = 0
    days_trend = max(Ca, Cb)
    # 5) factor
    factor = (0.0 if days_trend == 0 else
              0.1 if days_trend == 1 else
              0.2 if days_trend == 2 else 1.0)
    # 6) PID
    prev_err = state['prev_error']
    if days_trend == 1 or prev_err * err < 0:
        I = 0.0
    else:
        I = state['integral']
    I += err
    rawD = err - prev_err
    D    = DERIV_ALPHA * rawD + (1 - DERIV_ALPHA) * state['prev_derivative']
    Ki_mod = Ki * (3.0 if days_trend >= 3 else 1.0)
    delta = (Kp * err + Ki_mod * I + Kd * D) * factor
    # 7) price update
    FS = 1.0 if days_trend >= 3 else FAR_SMOOTH
    NS = 1.0 if days_trend >= 3 else NEAR_SMOOTH
    damp = lookup(DAMP_TABLE, on)
    new_far  = pf * (1 + (FS * delta) / 100.0)
    new_near = pn * (1 + (NS * damp * delta) / 100.0)
    # 8) clamps
    new_far  = min(mx_f, max(mn_f, new_far))
    new_near = min(mx_n, max(mn_n, new_near))
    # 9) persist
    state.update(
        near=new_near, far=new_far,
        demand=raw_dem, occ_near=on, occ_far=of,
        min_near=mn_n, max_near=mx_n, min_far=mn_f, max_far=mx_f,
        integral=I, prev_error=err, prev_derivative=D,
        consec_above=Ca, consec_below=Cb
    )
    save_state(state)
    return new_near, new_far, sm, err, days_trend, factor

# ─────────────────────────────────────────────────────────────────────────────
#  GUI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

class PriceBotApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PriceBot v6d")

        # burn-in any room that needs it
        for room in ROOMS:
            activate_room(room)
            perform_burn_in(self)

        # history for undo/redo
        self.history, self.future = [], []
        self.history.append({r: s.copy() for r, s in state_by_room.items()})

        self.build_ui()
        self.reload_ui()
        self.refresh_all_grey()

    # ─── UI BUILDERS ─────────────────────────────────────────────────────────
    def build_panel(self, parent, title):
        frm = ttk.LabelFrame(parent, text=title, padding=8)
        vars = {}

        # helper: place one row of old-value, entry, and label
        def add_field(row, label, key):
            vars['old_' + key] = tk.StringVar()
            ttk.Label(frm,
                      textvariable=vars['old_' + key],
                      foreground='grey') \
               .grid(row=row, column=0, sticky='w')
            vars[key] = tk.DoubleVar()
            ttk.Entry(frm,
                      textvariable=vars[key],
                      width=8) \
               .grid(row=row, column=1, sticky='w')
            ttk.Label(frm,
                      text=label) \
               .grid(row=row, column=2, sticky='w')

        # populate each field in its own row
        for r, (lbl, k) in enumerate([
            ("Now (€):",   "near"),
            ("Future (€):","far"),
            ("Demand (%):","demand"),
            ("Near Occ (%):","occ_near"),
            ("Far Occ (%):","occ_far"),
            ("Min Near (€):","min_near"),
            ("Max Near (€):","max_near"),
            ("Min Far (€):","min_far"),
            ("Max Far (€):","max_far")
        ]):
            add_field(r, lbl, k)

        # outputs for the new prices
        vars['out_near'] = ttk.Label(frm,
                                     text="New Near: —",
                                     font="TkDefaultFont 10 bold")
        vars['out_far']  = ttk.Label(frm,
                                     text="New Far:  —",
                                     font="TkDefaultFont 10 bold")
        vars['out_near'].grid(columnspan=3, pady=(10,0), sticky='w')
        vars['out_far'] .grid(columnspan=3, sticky='w')

        return frm, vars



    def build_ui(self):
        top = ttk.Frame(self, padding=10); top.pack(fill='x')
        self.panels = {}
        for i, name in enumerate(ROOMS):
            frm, vars = self.build_panel(top, name)
            frm.grid(row=0, column=i, padx=10, pady=5)
            self.panels[name] = vars
        ctrl = ttk.Frame(self); ctrl.pack(pady=10)
        ttk.Button(ctrl, text="◀ Back",    command=self.on_back)   .grid(row=0, column=0, padx=5)
        ttk.Button(ctrl, text="Run",       command=self.on_run)    .grid(row=0, column=1, padx=5)
        ttk.Button(ctrl, text="Forward ▶", command=self.on_forward).grid(row=0, column=2, padx=5)

    # ─── STATE → UI HELPERS ──────────────────────────────────────────────────
    def refresh_all_grey(self):
        for room, vars in self.panels.items():
            activate_room(room)
            for key, oldkey in [
                ('near','old_near'), ('far','old_far'),
                ('demand','old_demand'), ('occ_near','old_occ_near'),
                ('occ_far','old_occ_far'),
                ('min_near','old_min_near'), ('max_near','old_max_near'),
                ('min_far','old_min_far'),   ('max_far','old_max_far')
            ]:
                text = f"{state[key]:.2f}" + \
                       ("%" if "demand" in key or "occ" in key else "")
                vars[oldkey].set(text)

    def reload_ui(self):
        """Populate the white entry boxes from each room’s state."""
        for room, vars in self.panels.items():
            st = state_by_room[room]
            for key in (
                'near','far','demand','occ_near','occ_far',
                'min_near','max_near','min_far','max_far'
            ):
                vars[key].set(st[key])

    # ─── BUTTON CALLBACKS ────────────────────────────────────────────────────
    def on_run(self):
        vals_by_room = {}
        for room, vars in self.panels.items():
            try:
                vals_by_room[room] = [vars[k].get() for k in (
                    'near','far','demand','occ_near','occ_far',
                    'min_near','max_near','min_far','max_far'
                )]
            except tk.TclError:
                messagebox.showerror("Invalid input",
                                     "Fill all fields with numbers.")
                return

        # push undo snapshot
        self.history.append({r: s.copy() for r, s in state_by_room.items()})
        self.future.clear()

        # run engine per room
        for room, vals in vals_by_room.items():
            activate_room(room)
            nn, nf, *_ = compute_prices(*vals)
            pv = self.panels[room]
            pv['out_near'].config(text=f"New Near: €{nn:.2f}")
            pv['out_far'] .config(text=f"New Far:  €{nf:.2f}")
            save_state_for(room, state)

        self.refresh_all_grey()

    def on_back(self):
        if len(self.history) < 2:
            return
        self.future.append(self.history.pop())
        snap = self.history[-1]
        for room in ROOMS:
            state_by_room[room] = snap[room].copy()
        self.reload_ui()
        self.refresh_all_grey()

    def on_forward(self):
        if not self.future:
            return
        snap = self.future.pop()
        self.history.append(snap)
        for room in ROOMS:
            state_by_room[room] = snap[room].copy()
        self.reload_ui()
        self.refresh_all_grey()

if __name__ == "__main__":
    app = PriceBotApp()
    app.mainloop()
