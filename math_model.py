import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from math import pi

# Общие константы
R_Earth = 6371000  # Радиус Земли
G_M_Earth = 6.674 * 5.9722 * 10 ** 13  # Гравитационная постоянная * масса Земли
M = 0.029   # Молярная масса воздуха
R = 8.314   # Универсальная газовая постоянная

# Погодные условия на уровне моря
T0 = 19   # Температура , град Цельсия
P0 = 760  # Давление, мм ртутного столба

# Общие параметры ракеты
Cx = [[0, 0.165], [0.5, 0.149], [0.7, 0.175], [0.9, 0.255], [1, 0.304],
      [1.1, 0.36], [1.3, 0.484], [1.5, 0.5], [2, 0.51], [2.5, 0.502],
      [3, 0.5], [3.5, 0.485], [4, 0.463], [4.5, 0.458], [5, 0.447]]
S = 172  # Площадь миделя, м^2

# Время работы этапов, с
T1 = 123   # I ступень
T2 = 218   # II ступень
T3 = 242   # III ступень
T4 = 500   # Автономный полет
T5 = 270   # 1-й запуск РБ "Бриз-М"
T6 = 3000  # Автономный полет
T7 = 1080  # 2-й запуск РБ "Бриз-М"
T8 = 20000 # Автономный полет
T9 = 1200  # 3-й запуск РБ "Бриз-М"
T10 = 84000 # Автономный полет

# Массы ступеней, кг
M1 = 458.9 * 1000   # I ступень
M2 = 168.3 * 1000   # II ступень
M3 = 46.562 * 1000  # III ступень
M4_1 = 6.565 * 1000 # РБ "Бриз-М" (1-й этап)
M4_2 = 5.871 * 1000 # РБ "Бриз-М" (2-й этап)
M4_3 = 3.095 * 1000 # РБ "Бриз-М" (3-й этап)
M5 = 2210           # Спутник "Экран-М"


# Подбор коэффициента сопротивления по числу Маха (таблица Cx)
def get_cx(m):
    for i in range(len(Cx) - 1):
        if m == Cx[i][0]:
            return Cx[i][1]
        elif Cx[i][0] < m < Cx[i + 1][0]:
            return (Cx[i][1] + Cx[i + 1][1]) / 2
    return Cx[-1][1]


# Температура воздуха как функция высоты (упрощённая модель)
def get_temperature(h, T0):
    return max(h * (-0.0065) + T0, 4 - 273.15)


# Давление воздуха как функция высоты (барометрическая формула)
def get_pressure(h, p0):
    return (p0 * 133.32) * np.exp(-(M * 9.81 * h) /
                                  (R * (get_temperature(h, T0) + 273.15)))


# Плотность воздуха как функция высоты
def get_density(h):
    T = get_temperature(h, T0) + 273.15
    P = get_pressure(h, P0)
    return 0 if h >= 50000 else (P * M) / (R * T)


# Скорость звука как функция температуры
def get_speed_of_sound(t):
    return 250 if t < 150 else np.sqrt(1.4 * R * t / M)


# Расчёт аэродинамического сопротивления ракеты
def get_resistance(r, phi, r_dot, phi_dot):
    v2 = r_dot ** 2 + (r * phi_dot) ** 2
    mach = (v2 ** 0.5) / get_speed_of_sound(get_temperature(r - R_Earth, T0) + 273.15)
    return get_cx(mach) * get_density(r - R_Earth) * v2 * S / 2


# Перевод угла из градусов в радианы
def convert_to_rad(angle):
    return angle * pi / 180


# Этап разгона в атмосфере (с учётом тяги и сопротивления)
def acceleration_stage_atm(initial_conditions, T, F, sigma, M, k,
                           beta_start, beta_end):
    beta_incr = convert_to_rad((beta_end - beta_start) / T)
    beta_start = convert_to_rad(beta_start)

    def right_part(t, y):
        y1, y2, y3, y4 = y
        thrust_eff = (F + sigma * t) - get_resistance(y1, y3, y2, y4)
        return [
            y2,
            y1 * (y4 ** 2) - G_M_Earth / (y1 ** 2)
            + (np.cos(beta_start + beta_incr * t) / (M - k * t)) * thrust_eff,
            y4,
            (4000000 * np.sin(beta_start + beta_incr * t) * thrust_eff / (M - k * t)
             - 2 * y2 * y1 * y4) / (y1 ** 2)
        ]

    t = np.arange(0, T, 1)
    solver = solve_ivp(right_part, [0, T], initial_conditions,
                       method='RK45', dense_output=True)
    return solver.sol(t)


# Этап разгона в вакууме (без сопротивления атмосферы)
def acceleration_stage(initial_conditions, T, F, M, k,
                       beta_start, beta_end):
    beta_incr = convert_to_rad((beta_end - beta_start) / T)
    beta_start = convert_to_rad(beta_start)

    def right_part(t, y):
        y1, y2, y3, y4 = y
        return [
            y2,
            y1 * (y4 ** 2) - G_M_Earth / (y1 ** 2)
            + (np.cos(beta_start + beta_incr * t) / (M - k * t)) * F,
            y4,
            (4000000 * np.sin(beta_start + beta_incr * t) * F / (M - k * t)
             - 2 * y2 * y1 * y4) / (y1 ** 2)
        ]

    t = np.arange(0, T, 1)
    solver = solve_ivp(right_part, [0, T], initial_conditions,
                       method='RK45', dense_output=True)
    return solver.sol(t)


# Уравнения автономного полёта (без тяги и сопротивления)
def autonomous_flight(initial_conditions, T):
    def right_part(t, y):
        y1, y2, y3, y4 = y
        return [
            y2,
            y1 * (y4 ** 2) - G_M_Earth / (y1 ** 2),
            y4,
            -2 * ((y4 * y2) / y1)
        ]

    t = np.arange(0, T, 1)
    solver = solve_ivp(right_part, [0, T], initial_conditions,
                       method='RK45', dense_output=True)
    return solver.sol(t)


# Формирование траектории ракеты по всем этапам полёта
def get_vessel_trajectory(start_pos):
    traj = []
    traj.append(
        acceleration_stage_atm(start_pos, T1, 10026 * 1000, 7983.5,
                               M1 + M2 + M3 + M4_1 + M5, 3622, 0, 60))
    traj.append(
        acceleration_stage_atm(traj[-1][:, -1], T2, 2400 * 1000, 0,
                               M2 + M3 + M4_1 + M5, 731.63, 60, 60))
    traj.append(
        acceleration_stage(traj[-1][:, -1], T3, 583 * 1000,
                           M3 + M4_1 + M5, 180, 60, 60))
    traj.append(autonomous_flight(traj[-1][:, -1], T4))
    traj.append(
        acceleration_stage(traj[-1][:, -1], T5, 150 * 1000,
                           M4_1 + M5, 2.57, 60, 80))
    traj.append(autonomous_flight(traj[-1][:, -1], T6))
    traj.append(
        acceleration_stage(traj[-1][:, -1], T7, 32.2 * 1000,
                           M4_2 + M5, 2.57, 90, 90))
    traj.append(autonomous_flight(traj[-1][:, -1], T8))
    traj.append(
        acceleration_stage(traj[-1][:, -1], T9, 39.7 * 1000,
                           M4_3 + M5, 2.57, 89.6, 89.6))
    traj.append(autonomous_flight(traj[-1][:, -1], T10))
    return traj


# Объединение всех этапов в одну общую траекторию
def join_flight_stages(trajectory):
    total_T = T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10
    t = np.arange(0, total_T, 1)
    r = np.concatenate([stage[0, :] for stage in trajectory])
    r_dot = np.concatenate([stage[1, :] for stage in trajectory])
    phi = np.concatenate([stage[2, :] for stage in trajectory])
    phi_dot = np.concatenate([stage[3, :] for stage in trajectory])
    return t, np.array([r, r_dot, phi, phi_dot])


# Отрисовка графика высоты от времени
def draw_height(axis, trajectory):
    t, stages = join_flight_stages(trajectory)
    h = stages[0, :] - R_Earth
    axis.plot(t[:28000], h[:28000])
    axis.set_xlabel('t, с')
    axis.set_ylabel('h, м')
    axis.grid()


# Отрисовка графика скорости ракеты от времени
def draw_vessel_speed(axis, trajectory):
    t, stages = join_flight_stages(trajectory)
    v = np.sqrt(stages[1, :] ** 2 + (stages[0, :] * stages[3, :]) ** 2)
    axis.plot(t[:28000], v[:28000])
    axis.set_xlabel('t, с')
    axis.set_ylabel('V, м/с')
    axis.grid()


# Построение и вывод всех графиков полёта
def show_flight_parameter_plots(trajectory):
    fig, axs = plt.subplots(nrows=2, figsize=(8, 8))
    draw_height(axs[0], trajectory)
    draw_vessel_speed(axs[1], trajectory)
    plt.tight_layout()
    plt.show()


# ---- запуск расчёта и отрисовки ----
trajectory = get_vessel_trajectory([R_Earth, 0, 0, 0])
show_flight_parameter_plots(trajectory)