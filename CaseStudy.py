import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from skfuzzy import membership as mf

plt.close('all')
print("Matplotlib backend:", matplotlib.get_backend())

# Display toggles
SHOW_MEMBERSHIP = True
SHOW_AGGREGATED = True
SHOW_SURFACE_A  = True   # sleep vs work
SHOW_SURFACE_B  = True   # mood vs work

# Universes 
income   = ctrl.Antecedent(np.arange(0, 16000.1, 1.0), 'income_rm')
mood     = ctrl.Antecedent(np.arange(1, 6, 1), 'mood_5')         
sleep    = ctrl.Antecedent(np.arange(0, 12.01, 0.01), 'sleep_h')
work     = ctrl.Antecedent(np.arange(0, 80.1, 0.1), 'work_h_week')
caffeine = ctrl.Antecedent(np.arange(0, 400.1, 0.5), 'caffeine_mg')
stress   = ctrl.Consequent(np.arange(0, 100.1, 0.1), 'stress')

# Membership functions
# Income
income['Low']    = mf.trimf(income.universe, [0, 2625, 7875])
income['Medium'] = mf.trimf(income.universe, [1965, 8535, 15105])
income['High']   = mf.trimf(income.universe, [9796, 13844, 15867])

# Mood 
mood['Very Low']  = mf.trapmf(mood.universe, [1, 1, 1, 2])
mood['Low']       = mf.trimf(mood.universe,  [1, 2, 3])
mood['Neutral']   = mf.trimf(mood.universe,  [2, 3, 4])
mood['High']      = mf.trimf(mood.universe,  [3, 4, 5])
mood['Very High'] = mf.trapmf(mood.universe, [4, 5, 5, 5])

# Sleep (h/day)
sleep['Short']   = mf.trapmf(sleep.universe, [0, 0, 6, 7])
sleep['Optimal'] = mf.trapmf(sleep.universe, [6, 7, 9, 10])
sleep['Long']    = mf.trapmf(sleep.universe, [9, 10, 12, 12])

# Work (h/week)
work['Short']    = mf.trapmf(work.universe, [0, 0, 38, 40])
work['Optimal']  = mf.trapmf(work.universe, [38, 40, 45, 47])
work['Long']     = mf.trapmf(work.universe, [45, 47, 80, 80])

# Caffeine (mg/day)
caffeine['Low']      = mf.trapmf(caffeine.universe, [0, 0, 150, 200])
caffeine['Moderate'] = mf.trapmf(caffeine.universe, [150, 200, 300, 350])
caffeine['High']     = mf.trapmf(caffeine.universe, [300, 350, 400, 400])

# Stress (0–100)
stress['Low']        = mf.trapmf(stress.universe, [0, 0, 10, 25])
stress['Moderate']   = mf.trimf(stress.universe,  [20, 35, 50])
stress['High']       = mf.trimf(stress.universe,  [45, 60, 75])
stress['Very High']  = mf.trapmf(stress.universe, [70, 85, 100, 100])

# Rules 
rules = []
# AND rules
# Very High 
rules += [
    ctrl.Rule(work['Long'] & sleep['Short'] & mood['Very Low'], stress['Very High']),
    ctrl.Rule(work['Long'] & sleep['Short'] & caffeine['High'], stress['Very High']),
    ctrl.Rule(work['Long'] & mood['Very Low'] & income['Low'], stress['Very High']),
    ctrl.Rule(sleep['Short'] & mood['Very Low'] & income['Low'], stress['Very High']),
    ctrl.Rule(work['Long'] & sleep['Short'] & mood['Low'] & income['Low'], stress['Very High']),
    ctrl.Rule(work['Long'] & mood['Very Low'] & caffeine['High'], stress['Very High']),
    ctrl.Rule(sleep['Short'] & mood['Very Low'] & caffeine['High'], stress['Very High']),
    ctrl.Rule(work['Long'] & sleep['Short'] & mood['Very Low'] & caffeine['High'], stress['Very High']),
]

# High
rules += [
    ctrl.Rule(work['Long'] & sleep['Short'] & mood['Neutral'], stress['High']),
    ctrl.Rule(work['Long'] & mood['Low'] & caffeine['High'],   stress['High']),
    ctrl.Rule(sleep['Short'] & mood['Low'] & caffeine['High'], stress['High']),
    ctrl.Rule(work['Long'] & income['Low'] & mood['Neutral'],  stress['High']),
    ctrl.Rule(sleep['Short'] & income['Low'] & mood['Low'],    stress['High']),
    ctrl.Rule(work['Long'] & sleep['Optimal'] & mood['Very Low'],  stress['High']),
    ctrl.Rule(sleep['Short'] & work['Optimal'] & mood['Very Low'], stress['High']),
    ctrl.Rule(work['Long'] & caffeine['Moderate'] & mood['Low'],   stress['High']),
    ctrl.Rule(sleep['Short'] & caffeine['Moderate'] & mood['Low'], stress['High']),
    ctrl.Rule(income['Low'] & mood['Very Low'] & caffeine['High'], stress['High']),
]

# Moderate
rules += [
    ctrl.Rule(work['Long'] & sleep['Optimal'] & mood['Neutral'],  stress['Moderate']),
    ctrl.Rule(sleep['Short'] & work['Optimal'] & caffeine['Low'], stress['Moderate']),
    ctrl.Rule(mood['Low'] & work['Optimal'] & sleep['Optimal'],   stress['Moderate']),
    ctrl.Rule(income['Low'] & mood['Neutral'] & work['Optimal'],  stress['Moderate']),
    ctrl.Rule(work['Short'] & mood['Low'] & caffeine['Moderate'], stress['Moderate']),
    ctrl.Rule(sleep['Short'] & mood['Neutral'] & caffeine['Low'], stress['Moderate']),
    ctrl.Rule(work['Long'] & caffeine['Low'] & mood['Neutral'],   stress['Moderate']),
]

# Low
rules += [
    ctrl.Rule(sleep['Optimal'] & work['Optimal'] & caffeine['Low'] & mood['High'], stress['Low']),
    ctrl.Rule(sleep['Long'] & work['Short'] & mood['Neutral'], stress['Low']),
    ctrl.Rule(income['High'] & sleep['Optimal'] & mood['High'], stress['Low']),
    ctrl.Rule(work['Optimal'] & mood['Very High'] & caffeine['Low'], stress['Low']),
    ctrl.Rule(sleep['Optimal'] & work['Short'] & mood['High'], stress['Low']),
]

# OR rules (|) 
rules.append(ctrl.Rule((work['Long'] & sleep['Short']) | (caffeine['High'] & mood['Very Low']), stress['Very High']))
rules.append(ctrl.Rule(work['Long'] | ((sleep['Short'] & caffeine['Moderate']) & (mood['Low'] | mood['Neutral'])), stress['High']))
rules.append(ctrl.Rule((sleep['Optimal'] | (work['Short'] & caffeine['Low'])) & (mood['High'] | mood['Very High']), stress['Low']))
rules.append(ctrl.Rule((income['Low'] | mood['Low']) & work['Optimal'], stress['Moderate']))

# Build controller
system = ctrl.ControlSystem(rules)

# Helpers
def _round_mood_int(m):
    return int(min(5, max(1, round(float(m)))))

def _deg(var, term, x):
    return float(fuzz.interp_membership(var.universe, var[term].mf, x))

def analyze_drivers(income_rm_val, mood_1to5, sleep_h_val, work_h_week_val, caffeine_mg_val):
    m_int = _round_mood_int(mood_1to5)
    drivers = {
        'work_long': _deg(work, 'Long', work_h_week_val),
        'sleep_short': _deg(sleep, 'Short', sleep_h_val),
        'caff_high': _deg(caffeine, 'High', caffeine_mg_val),
        'caff_moderate': _deg(caffeine, 'Moderate', caffeine_mg_val),
        'mood_vlow': _deg(mood, 'Very Low', m_int),
        'mood_low': _deg(mood, 'Low', m_int),
        'income_low': _deg(income, 'Low', income_rm_val),
        'sleep_optimal': _deg(sleep, 'Optimal', sleep_h_val),
        'work_optimal': _deg(work, 'Optimal', work_h_week_val),
        'work_short': _deg(work, 'Short', work_h_week_val),
        'mood_high': _deg(mood, 'High', m_int),
        'mood_vhigh': _deg(mood, 'Very High', m_int),
        'caff_low': _deg(caffeine, 'Low', caffeine_mg_val),
        'income_high': _deg(income, 'High', income_rm_val),
    }
    return drivers, m_int

def recommend(actions_needed, label):
    recs = []

    if label in ('Very High', 'High'):
        if actions_needed.get('sleep_short', 0) > 0.4:
            recs.append("Prioritise sleep tonight.")
        if actions_needed.get('work_long', 0) > 0.4:
            recs.append("Reduce workload for 24 hour; delay non-urgent tasks.")
        if actions_needed.get('mood_vlow', 0) > 0.4 or actions_needed.get('mood_low', 0) > 0.5:
            recs.append("Do a 2-min breathing reset, then a 10-min walk.")
        if actions_needed.get('caff_high', 0) > 0.4 or actions_needed.get('caff_moderate', 0) > 0.6:
            recs.append("Cap caffeine ≤200 mg today; avoid after 4 pm.")
        if actions_needed.get('income_low', 0) > 0.5:
            recs.append("Schedule a budgeting check-in this week.")
        if not recs:
            recs.append("Block 30–60 min for rest, hydration, and light movement.")
    elif label == 'Moderate':
        if actions_needed.get('sleep_optimal', 0) < 0.4 and actions_needed.get('sleep_short', 0) > 0.2:
            recs.append("Extend sleep by 30–60 min for two nights.")
        if actions_needed.get('work_long', 0) > 0.3:
            recs.append("Work in 50/10 focus–break cycles.")
        if actions_needed.get('caff_moderate', 0) > 0.4:
            recs.append("Swap the next coffee for water.")
        if actions_needed.get('mood_low', 0) > 0.4:
            recs.append("Take a 10-min unwind break or brief walk.")
        if not recs:
            recs.append("Keep routines steady today; review tomorrow.")
    else:  
        if actions_needed.get('sleep_optimal', 0) > 0.5 and actions_needed.get('work_optimal', 0) > 0.4:
            recs.append("Good balance—keep your current rhythm.")
        else:
            recs.append("Maintain habits; add a short daily walk.")

    return recs[:5]

def predict_stress_with_recs(income_rm_val, mood_1to5, sleep_h_val, work_h_week_val, caffeine_mg_val):
    m_int = _round_mood_int(mood_1to5)
    sim = ctrl.ControlSystemSimulation(system)
    sim.input['income_rm']   = float(income_rm_val)
    sim.input['mood_5']      = float(m_int)
    sim.input['sleep_h']     = float(sleep_h_val)
    sim.input['work_h_week'] = float(work_h_week_val)
    sim.input['caffeine_mg'] = float(caffeine_mg_val)
    sim.compute()

    score = float(sim.output['stress'])
    label = 'Low' if score < 25 else 'Moderate' if score < 50 else 'High' if score < 75 else 'Very High'

    drv, m_int_used = analyze_drivers(income_rm_val, m_int, sleep_h_val, work_h_week_val, caffeine_mg_val)
    recs = recommend(drv, label)
    return score, label, recs, sim, m_int_used

# Case Study 
cases = [
    ('Case A', 1800, 2, 5.0, 60, 350),
    ('Case B', 6000, 3, 7.5, 42, 120),
    ('Case C', 12000, 4, 9.0, 30, 50),
    ('Case D', 3500, 2, 6.0, 50, 280),
]
for name, inc, md, slp, wk, caf in cases:
    score, label, recs, _, mood_used = predict_stress_with_recs(inc, md, slp, wk, caf)
    print(f"\n{name}: Stress={score:.1f} → {label} "
          f"(income=RM {inc}, mood={mood_used}/5, sleep={slp} h, work={wk} h/week, caffeine={caf} mg)")
    print("  Recommendations:")
    for r in recs:
        print("   -", r)

# Membership diagrams
if SHOW_MEMBERSHIP:
    income.view()

    mood.view()
    plt.gca().set_xticks([1,2,3,4,5])  

    sleep.view()
    work.view()
    caffeine.view()
    stress.view()
    plt.gca().set_xticks(np.arange(0, 101, 20))
    plt.show()

# Aggregated output
if SHOW_AGGREGATED:
    _, _, _, sim_plot, _ = predict_stress_with_recs(3000, 3, 5.5, 50, 180)
    stress.view(sim=sim_plot)
    plt.title("Aggregated output for example inputs")
    plt.show()

# Output surfaces
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
            sim_local.input['mood_5']      = float(_round_mood_int(fixed.get('mood_5', 3)))
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

# Surface A: Sleep vs Work 
if SHOW_SURFACE_A:
    XA, YA, ZA = output_space(
        var_x='sleep_h', var_y='work_h_week',
        fixed={'income_rm': 5000.0, 'mood_5': 3, 'caffeine_mg': 150.0,
               'x_univ': sleep.universe, 'y_univ': work.universe}
    )
    plot3d(XA, YA, ZA, 'Stress surface: Sleep vs Work', 'sleep (h/day)', 'work (h/week)')

# Surface B: Mood vs Work 
if SHOW_SURFACE_B:
    mood_vals = np.arange(1, 6, 1)
    work_vals = np.linspace(work.universe.min(), work.universe.max(), 60)
    X_mood, Y_work = np.meshgrid(mood_vals, work_vals)
    Z = np.zeros_like(X_mood, dtype=float)

    sim_local = ctrl.ControlSystemSimulation(system)
    for i in range(X_mood.shape[0]):
        for j in range(X_mood.shape[1]):
            sim_local.input['income_rm']   = 3000.0
            sim_local.input['sleep_h']     = 6.5
            sim_local.input['caffeine_mg'] = 120.0
            sim_local.input['mood_5']      = float(int(X_mood[i, j]))
            sim_local.input['work_h_week'] = float(Y_work[i, j])
            try:
                sim_local.compute()
                Z[i, j] = sim_local.output['stress']
            except Exception:
                Z[i, j] = np.nan

    plot3d(X_mood, Y_work, Z, 'Stress surface: Mood vs Work', 'mood (1–5)', 'work (h/week)')
