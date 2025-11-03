import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from skfuzzy import membership as mf

# Clear stale figures (handy for Spyder)
plt.close('all')
print("Matplotlib backend:", matplotlib.get_backend())

# -----------------------------
# Display toggles (ON)
SHOW_MEMBERSHIP = True     # show membership function diagrams
SHOW_AGGREGATED = True     # show aggregated output curve for an example run
SHOW_SURFACE_A  = True     # 3D surface: sleep vs work
SHOW_SURFACE_B  = True     # 3D surface: mood vs work
# -----------------------------

# ---------- 1) Universes ----------
# Income High triangle exceeds RM 10,000, so extend to 16,000 to honour your fit vector.
income   = ctrl.Antecedent(np.arange(0, 16000.1, 1.0), 'income_rm')
mood     = ctrl.Antecedent(np.arange(1, 5.01, 0.01),    'mood_5')
sleep    = ctrl.Antecedent(np.arange(0, 12.01, 0.01),   'sleep_h')
work     = ctrl.Antecedent(np.arange(0, 80.1, 0.1),     'work_h_week')
caffeine = ctrl.Antecedent(np.arange(0, 400.1, 0.5),    'caffeine_mg')

stress   = ctrl.Consequent(np.arange(0, 100.1, 0.1),    'stress')

# ---------- 2) Membership functions  ----------
# Income (triangles)
income['Low']    = mf.trimf(income.universe,  [0, 2625, 7875])
income['Medium'] = mf.trimf(income.universe,  [1965, 8535, 15105])
income['High']   = mf.trimf(income.universe,  [9796, 13844, 15867])

# Mood (1–5): five terms (smooth overlaps)
mood['Very Low']  = mf.trapmf(mood.universe,  [1.00, 1.00, 1.50, 2.00])
mood['Low']       = mf.trimf(mood.universe,   [1.70, 2.25, 2.80])
mood['Neutral']   = mf.trimf(mood.universe,   [2.50, 3.00, 3.50])
mood['High']      = mf.trimf(mood.universe,   [3.20, 3.80, 4.40])
mood['Very High'] = mf.trapmf(mood.universe,  [4.00, 4.50, 5.00, 5.00])

# Sleep (hours/day)
sleep['Short']   = mf.trapmf(sleep.universe,  [0, 0, 6, 7])
sleep['Optimal'] = mf.trapmf(sleep.universe,  [6, 7, 9, 10])
sleep['Long']    = mf.trapmf(sleep.universe,  [9, 10, 12, 12])

# Work (hours/week)
work['Short']    = mf.trapmf(work.universe,   [0, 0, 38, 40])
work['Optimal']  = mf.trapmf(work.universe,   [38, 40, 45, 47])
work['Long']     = mf.trapmf(work.universe,   [45, 47, 80, 80])

# Caffeine (mg/day)
caffeine['Low']      = mf.trapmf(caffeine.universe, [0, 0, 150, 200])
caffeine['Moderate'] = mf.trapmf(caffeine.universe, [150, 200, 300, 350])
caffeine['High']     = mf.trapmf(caffeine.universe, [300, 350, 400, 400])

# Output: Stress (0–100)
stress['Low']        = mf.trapmf(stress.universe, [0, 0, 10, 25])
stress['Moderate']   = mf.trimf(stress.universe,  [20, 35, 50])
stress['High']       = mf.trimf(stress.universe,  [45, 60, 75])
stress['Very High']  = mf.trapmf(stress.universe, [70, 85, 100, 100])

# ---------- 3) Rule base (2.2.4; no weight keyword) ----------
rules = []

# Very High (compound evidence)
rules += [
    ctrl.Rule(work['Long'] & sleep['Short'] & mood['Very Low'],                     stress['Very High']),
    ctrl.Rule(work['Long'] & sleep['Short'] & caffeine['High'],                     stress['Very High']),
    ctrl.Rule(work['Long'] & mood['Very Low'] & income['Low'],                      stress['Very High']),
    ctrl.Rule(sleep['Short'] & mood['Very Low'] & income['Low'],                    stress['Very High']),
    ctrl.Rule(work['Long'] & sleep['Short'] & mood['Low'] & income['Low'],          stress['Very High']),
    ctrl.Rule(work['Long'] & mood['Very Low'] & caffeine['High'],                   stress['Very High']),
    ctrl.Rule(sleep['Short'] & mood['Very Low'] & caffeine['High'],                 stress['Very High']),
    ctrl.Rule(work['Long'] & sleep['Short'] & mood['Very Low'] & caffeine['High'],  stress['Very High']),
]

# High (strong but not extreme)
rules += [
    ctrl.Rule(work['Long'] & sleep['Short'] & mood['Neutral'],                      stress['High']),
    ctrl.Rule(work['Long'] & mood['Low'] & caffeine['High'],                        stress['High']),
    ctrl.Rule(sleep['Short'] & mood['Low'] & caffeine['High'],                      stress['High']),
    ctrl.Rule(work['Long'] & income['Low'] & mood['Neutral'],                       stress['High']),
    ctrl.Rule(sleep['Short'] & income['Low'] & mood['Low'],                         stress['High']),
    ctrl.Rule(work['Long'] & sleep['Optimal'] & mood['Very Low'],                   stress['High']),
    ctrl.Rule(sleep['Short'] & work['Optimal'] & mood['Very Low'],                  stress['High']),
    ctrl.Rule(work['Long'] & caffeine['Moderate'] & mood['Low'],                    stress['High']),
    ctrl.Rule(sleep['Short'] & caffeine['Moderate'] & mood['Low'],                  stress['High']),
    ctrl.Rule(income['Low'] & mood['Very Low'] & caffeine['High'],                  stress['High']),
]

# Moderate (single primary or mixed signals)
rules += [
    ctrl.Rule(work['Long'] & sleep['Optimal'] & mood['Neutral'],                    stress['Moderate']),
    ctrl.Rule(sleep['Short'] & work['Optimal'] & caffeine['Low'],                   stress['Moderate']),
    ctrl.Rule(mood['Low'] & work['Optimal'] & sleep['Optimal'],                     stress['Moderate']),
    ctrl.Rule(income['Low'] & mood['Neutral'] & work['Optimal'],                    stress['Moderate']),
    ctrl.Rule(work['Short'] & mood['Low'] & caffeine['Moderate'],                   stress['Moderate']),
    ctrl.Rule(sleep['Short'] & mood['Neutral'] & caffeine['Low'],                   stress['Moderate']),
    ctrl.Rule(work['Long'] & caffeine['Low'] & mood['Neutral'],                     stress['Moderate']),
]

# Low (protective patterns)
rules += [
    ctrl.Rule(sleep['Optimal'] & work['Optimal'] & caffeine['Low'] & mood['High'],  stress['Low']),
    ctrl.Rule(sleep['Long'] & work['Short'] & mood['Neutral'],                       stress['Low']),
    ctrl.Rule(income['High'] & sleep['Optimal'] & mood['High'],                      stress['Low']),
    ctrl.Rule(work['Optimal'] & mood['Very High'] & caffeine['Low'],                 stress['Low']),
    ctrl.Rule(sleep['Optimal'] & work['Short'] & mood['High'],                       stress['Low']),
]

# ---------- 4) Build controller ----------
system = ctrl.ControlSystem(rules)

# ---------- 5) Helpers: prediction, driver analysis, recommendations ----------

def _deg(var, term, x):
    """Membership degree of x in var[term]."""
    return float(fuzz.interp_membership(var.universe, var[term].mf, x))

def analyze_drivers(income_rm_val, mood_1to5, sleep_h_val, work_h_week_val, caffeine_mg_val):
    """Return key driver scores (0..1) indicating what's pushing risk up/down."""
    drivers = {
        # risk-raising signals
        'work_long': _deg(work, 'Long', work_h_week_val),
        'sleep_short': _deg(sleep, 'Short', sleep_h_val),
        'caff_high': _deg(caffeine, 'High', caffeine_mg_val),
        'caff_moderate': _deg(caffeine, 'Moderate', caffeine_mg_val),
        'mood_vlow': _deg(mood, 'Very Low', mood_1to5),
        'mood_low': _deg(mood, 'Low', mood_1to5),
        'income_low': _deg(income, 'Low', income_rm_val),

        # protective signals
        'sleep_optimal': _deg(sleep, 'Optimal', sleep_h_val),
        'work_optimal': _deg(work, 'Optimal', work_h_week_val),
        'work_short': _deg(work, 'Short', work_h_week_val),
        'mood_high': _deg(mood, 'High', mood_1to5),
        'mood_vhigh': _deg(mood, 'Very High', mood_1to5),
        'caff_low': _deg(caffeine, 'Low', caffeine_mg_val),
        'income_high': _deg(income, 'High', income_rm_val),
    }
    return drivers

def recommend(actions_needed, label):
    """Turn driver flags into actionable recommendations."""
    recs = []

    # Strong actions for Very High / High
    if label in ('Very High', 'High'):
        if actions_needed.get('sleep_short', 0) > 0.4:
            recs.append("Prioritise sleep tonight: aim 7–9 hours; screens off 60 minutes before bed.")
        if actions_needed.get('work_long', 0) > 0.4:
            recs.append("Reduce workload for 24 hours. Use 50/10 focus–break cycles; postpone non-urgent tasks.")
        if actions_needed.get('mood_vlow', 0) > 0.4 or actions_needed.get('mood_low', 0) > 0.5:
            recs.append("Do a 2-minute 4–7–8 breathing reset, then a 10-minute walk or check-in with a friend.")
        if actions_needed.get('caff_high', 0) > 0.4 or actions_needed.get('caff_moderate', 0) > 0.6:
            recs.append("Cut caffeine now; keep under ~200 mg today and avoid after 4 pm.")
        if actions_needed.get('income_low', 0) > 0.5:
            recs.append("Schedule a budgeting check-in or look at campus financial aid resources this week.")
        if not recs:
            recs.append("Create a recovery block: 30–60 minutes for rest, hydration, and light movement. Reassess after.")

    # Gentle tuning for Moderate
    if label == 'Moderate':
        if actions_needed.get('sleep_optimal', 0) < 0.4 and actions_needed.get('sleep_short', 0) > 0.2:
            recs.append("Extend sleep by 30–60 minutes for the next two nights.")
        if actions_needed.get('work_long', 0) > 0.3:
            recs.append("Break study/work into smaller blocks and insert a short walk every hour.")
        if actions_needed.get('caff_moderate', 0) > 0.4:
            recs.append("Swap the next coffee for water or decaf.")
        if actions_needed.get('mood_low', 0) > 0.4:
            recs.append("Take a 10-minute relaxation break or message someone you trust.")
        if not recs:
            recs.append("Keep routines steady today; review sleep and workload tomorrow.")

    # Maintenance for Low
    if label == 'Low':
        if actions_needed.get('sleep_optimal', 0) > 0.5 and actions_needed.get('work_optimal', 0) > 0.4:
            recs.append("Good balance — keep your sleep schedule and work rhythm as is.")
        else:
            recs.append("Maintain current habits; add a short walk and steady meals to keep risk low.")

    if len(recs) > 5:
        recs = recs[:5]
    return recs

def predict_stress_with_recs(income_rm_val, mood_1to5, sleep_h_val, work_h_week_val, caffeine_mg_val):
    """Compute stress score/label + driver analysis + recommendations."""
    sim = ctrl.ControlSystemSimulation(system)  # fresh sim each call
    sim.input['income_rm']   = float(income_rm_val)
    sim.input['mood_5']      = float(mood_1to5)
    sim.input['sleep_h']     = float(sleep_h_val)
    sim.input['work_h_week'] = float(work_h_week_val)
    sim.input['caffeine_mg'] = float(caffeine_mg_val)
    sim.compute()

    score = float(sim.output['stress'])
    if score < 25: label = 'Low'
    elif score < 50: label = 'Moderate'
    elif score < 75: label = 'High'
    else: label = 'Very High'

    drv = analyze_drivers(income_rm_val, mood_1to5, sleep_h_val, work_h_week_val, caffeine_mg_val)

    ups = {
        'Work: Long': drv['work_long'],
        'Sleep: Short': drv['sleep_short'],
        'Caffeine: High': drv['caff_high'],
        'Caffeine: Moderate': drv['caff_moderate'],
        'Mood: Very Low': drv['mood_vlow'],
        'Mood: Low': drv['mood_low'],
        'Income: Low': drv['income_low'],
    }
    downs = {
        'Sleep: Optimal': drv['sleep_optimal'],
        'Work: Optimal': drv['work_optimal'],
        'Work: Short': drv['work_short'],
        'Mood: High/Very High': max(drv['mood_high'], drv['mood_vhigh']),
        'Caffeine: Low': drv['caff_low'],
        'Income: High': drv['income_high'],
    }

    def top_items(d, thresh=0.35, k=5):
        return [(k_, round(v_, 2)) for k_, v_ in sorted(d.items(), key=lambda t: t[1], reverse=True) if v_ > thresh][:k]

    top_up = top_items(ups, 0.35, 5)
    top_down = top_items(downs, 0.45, 3)
    recs = recommend(drv, label)

    return score, label, top_up, top_down, recs, sim

# ---------- 6) Demo: run cases, print recommendations ----------
cases = [
    ('Case A', 1800, 1.8, 5.0, 60, 350),   # likely Very High
    ('Case B', 6000, 3.0, 7.5, 42, 120),   # likely Moderate
    ('Case C', 12000, 4.2, 9.0, 30,  50),  # likely Low
    ('Case D', 3500, 2.4, 6.0, 50, 280),   # likely High
]
for name, inc, md, slp, wk, caf in cases:
    score, label, top_up, top_down, recs, _ = predict_stress_with_recs(inc, md, slp, wk, caf)
    print(f"\n{name}: Stress={score:.1f} → {label} "
          f"(income={inc} RM, mood={md}/5, sleep={slp}h, work={wk}h/wk, caffeine={caf}mg)")
    if top_up:
        print("  Drivers pushing risk up:", "; ".join([f"{k} {v}" for k, v in top_up]))
    if top_down:
        print("  Protective factors:", "; ".join([f"{k} {v}" for k, v in top_down]))
    if recs:
        print("  Recommendations:")
        for r in recs:
            print("   -", r)

# ---------- 7) Membership diagrams ----------
if SHOW_MEMBERSHIP:
    income.view(); mood.view(); sleep.view(); work.view(); caffeine.view(); stress.view()
    plt.show()

# ---------- 8) Aggregated output curve for an example ----------
if SHOW_AGGREGATED:
    _, _, _, _, _, sim_plot = predict_stress_with_recs(3000, 2.8, 5.5, 50, 180)
    stress.view(sim=sim_plot)
    plt.title("Aggregated output for example inputs")
    plt.show()

# ---------- 9) Lab-2 style output surfaces ----------
def plot3d(X, Y, Z, title, xlabel, ylabel):
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis',
                           linewidth=0.4, antialiased=True)
    zoff = np.nanmin(Z) - 5 if np.isfinite(Z).any() else -2.5
    ax.contourf(X, Y, Z, zdir='z', offset=zoff, cmap='viridis', alpha=0.5)
    ax.contourf(X, Y, Z, zdir='x', offset=X.max()*1.1, cmap='viridis', alpha=0.5)
    ax.contourf(X, Y, Z, zdir='y', offset=Y.max()*1.1, cmap='viridis', alpha=0.5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel('stress')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.6, aspect=10)
    ax.view_init(30, 200)
    plt.show()

def output_space(var_x, var_y, fixed):
    sim_local = ctrl.ControlSystemSimulation(system)
    X, Y = np.meshgrid(
        np.linspace(fixed['x_univ'].min(), fixed['x_univ'].max(), 60),
        np.linspace(fixed['y_univ'].min(), fixed['y_univ'].max(), 60)
    )
    Z = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            sim_local.input['income_rm']   = fixed.get('income_rm', 5000.0)
            sim_local.input['mood_5']      = fixed.get('mood_5', 3.0)
            sim_local.input['sleep_h']     = fixed.get('sleep_h', 6.5)
            sim_local.input['work_h_week'] = fixed.get('work_h_week', 42.0)
            sim_local.input['caffeine_mg'] = fixed.get('caffeine_mg', 150.0)
            sim_local.input[var_x]         = float(X[i, j])
            sim_local.input[var_y]         = float(Y[i, j])
            try:
                sim_local.compute()
                Z[i, j] = sim_local.output['stress']
            except Exception:
                Z[i, j] = np.nan
    return X, Y, Z

# Surface A: Sleep vs Work (others fixed)
if SHOW_SURFACE_A:
    XA, YA, ZA = output_space(
        var_x='sleep_h', var_y='work_h_week',
        fixed={'income_rm': 5000.0, 'mood_5': 3.0, 'caffeine_mg': 150.0,
               'x_univ': sleep.universe, 'y_univ': work.universe}
    )
    plot3d(XA, YA, ZA, 'Stress surface: Sleep vs Work',
           'sleep (h/day)', 'work (h/week)')

# Surface B: Mood vs Work (others fixed)
if SHOW_SURFACE_B:
    XB, YB, ZB = output_space(
        var_x='mood_5', var_y='work_h_week',
        fixed={'income_rm': 3000.0, 'sleep_h': 6.5, 'caffeine_mg': 120.0,
               'x_univ': mood.universe,  'y_univ': work.universe}
    )
    plot3d(XB, YB, ZB, 'Stress surface: Mood vs Work',
           'mood (1–5)', 'work (h/week)')
