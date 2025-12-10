import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from math import pi

# Глобальные константы

# Параметры планеты
R_PLANET = 6371000               # Радиус Земли, м
MU_PLANET = 6.674 * 5.9722 * 10**13  # Гравитационная постоянная * масса Земли

# Газовые константы
AIR_MOLAR_MASS = 0.029   # Молярная масса воздуха, кг/моль
GAS_CONST = 8.314        # Универсальная газовая постоянная, Дж/(моль*К)

# Погодные условия на уровне моря
T_SEA_LEVEL = 19   # Температура на старте, °C
P_SEA_LEVEL = 760  # Давление, мм ртутного столба

# Аэродинамические параметры ракеты
CX_TABLE = [
    [0,   0.165], [0.5, 0.149], [0.7, 0.175], [0.9, 0.255], [1,   0.304],
    [1.1, 0.36 ], [1.3, 0.484], [1.5, 0.5  ], [2,   0.51 ], [2.5, 0.502],
    [3,   0.5  ], [3.5, 0.485], [4,   0.463], [4.5, 0.458], [5,   0.447]
]
REF_AREA = 172  # Площадь миделя, м^2

# Время работы этапов, с
STAGE1_TIME    = 123    # I ступень
STAGE2_TIME    = 218     # II ступень
STAGE3_TIME    = 242     # III ступень
COAST1_TIME    = 500     # Автономный полет
BREEZE1_TIME   = 270     # 1-й запуск РБ "Бриз-М"
COAST2_TIME    = 3000    # Автономный полет
BREEZE2_TIME   = 1080    # 2-й запуск РБ "Бриз-М"
COAST3_TIME    = 20000    # Автономный полет
BREEZE3_TIME   = 1200    # 3-й запуск РБ "Бриз-М"
COAST4_TIME    = 84000   # Автономный полет

# Массы ступеней, кг
MASS_STAGE1   = 458.9 * 1000   # I ступень
MASS_STAGE2   = 168.3 * 1000   # II ступень
MASS_STAGE3   = 46.562 * 1000   # III ступень
MASS_BREEZE1  = 6.565 * 1000   # РБ "Бриз-М" (1-й этап)
MASS_BREEZE2  = 5.871 * 1000   # РБ "Бриз-М" (2-й этап)
MASS_BREEZE3  = 3.095 * 1000   # РБ "Бриз-М" (3-й этап)
MASS_PAYLOAD  = 2210         # Спутник "Экран-М"


# Вспомогательные функции

def deg_to_rad(angle_deg):
    """Переводит угол из градусов в радианы."""
    return angle_deg * pi / 180


def lookup_drag_coefficient(mach):
    """Находит коэффициент лобового сопротивления по числу Маха по таблице."""
    for i in range(len(CX_TABLE) - 1):
        if mach == CX_TABLE[i][0]:
            return CX_TABLE[i][1]
        elif CX_TABLE[i][0] < mach < CX_TABLE[i + 1][0]:
            return (CX_TABLE[i][1] + CX_TABLE[i + 1][1]) / 2
    return CX_TABLE[-1][1]


# Атмосфера и аэродинамика

def temperature_profile(h, t0_celsius):
    """Температура воздуха как функция высоты (упрощённая модель)."""
    return max(h * (-0.0065) + t0_celsius, 4 - 273.15)


def pressure_profile(h, p0_mm):
    """Давление воздуха как функция высоты (барометрическая формула)."""
    return (p0_mm * 133.32) * np.exp(
        -(AIR_MOLAR_MASS * 9.81 * h) /
        (GAS_CONST * (temperature_profile(h, T_SEA_LEVEL) + 273.15))
    )


def air_density(h):
    """Плотность воздуха как функция высоты."""
    T = temperature_profile(h, T_SEA_LEVEL) + 273.15
    P = pressure_profile(h, P_SEA_LEVEL)
    return 0 if h >= 50000 else (P * AIR_MOLAR_MASS) / (GAS_CONST * T)


def sound_speed(temp_kelvin):
    """Скорость звука как функция температуры."""
    return 250 if temp_kelvin < 150 else np.sqrt(1.4 * GAS_CONST * temp_kelvin / AIR_MOLAR_MASS)


def drag_force(r, phi, r_dot, phi_dot):
    """Расчёт аэродинамической силы сопротивления ракеты."""
    v2 = r_dot**2 + (r * phi_dot)**2
    temp = temperature_profile(r - R_PLANET, T_SEA_LEVEL) + 273.15
    mach = np.sqrt(v2) / sound_speed(temp)
    cx = lookup_drag_coefficient(mach)
    rho = air_density(r - R_PLANET)
    return cx * rho * v2 * REF_AREA / 2


# Интегрирование движения по этапам

def integrate_thrust_atmosphere(initial_state, duration, thrust_start, thrust_slope,
                                mass_start, mass_flow_rate, beta_start_deg, beta_end_deg):
    """Разгонный участок в атмосфере: учитывается тяга и аэродинамическое сопротивление."""
    beta_step = deg_to_rad((beta_end_deg - beta_start_deg) / duration)
    beta0 = deg_to_rad(beta_start_deg)

    def rhs(t, y):
        r, r_dot, phi, phi_dot = y
        effective_thrust = (thrust_start + thrust_slope * t) - drag_force(r, phi, r_dot, phi_dot)
        radial_acc = (
            r * phi_dot**2
            - MU_PLANET / (r**2)
            + (np.cos(beta0 + beta_step * t) / (mass_start - mass_flow_rate * t)) * effective_thrust
        )
        angular_acc = (
            4000000 * np.sin(beta0 + beta_step * t) * effective_thrust / (mass_start - mass_flow_rate * t)
            - 2 * r_dot * r * phi_dot
        ) / (r**2)

        return [r_dot, radial_acc, phi_dot, angular_acc]

    t_eval = np.arange(0, duration, 1)
    solution = solve_ivp(rhs, [0, duration], initial_state, method='RK45', dense_output=True)
    return solution.sol(t_eval)


def integrate_thrust_vacuum(initial_state, duration, thrust, mass_start,
                            mass_flow_rate, beta_start_deg, beta_end_deg):
    """Разгонный участок в вакууме: учитывается только тяга, без сопротивления атмосферы."""
    beta_step = deg_to_rad((beta_end_deg - beta_start_deg) / duration)
    beta0 = deg_to_rad(beta_start_deg)

    def rhs(t, y):
        r, r_dot, phi, phi_dot = y
        radial_acc = (
            r * phi_dot**2
            - MU_PLANET / (r**2)
            + (np.cos(beta0 + beta_step * t) / (mass_start - mass_flow_rate * t)) * thrust
        )
        angular_acc = (
            4000000 * np.sin(beta0 + beta_step * t) * thrust / (mass_start - mass_flow_rate * t)
            - 2 * r_dot * r * phi_dot
        ) / (r**2)

        return [r_dot, radial_acc, phi_dot, angular_acc]

    t_eval = np.arange(0, duration, 1)
    solution = solve_ivp(rhs, [0, duration], initial_state, method='RK45', dense_output=True)
    return solution.sol(t_eval)


def integrate_coasting(initial_state, duration):
    """Участок автономного полёта: только гравитация, без тяги и сопротивления."""
    def rhs(t, y):
        r, r_dot, phi, phi_dot = y
        radial_acc = r * phi_dot**2 - MU_PLANET / (r**2)
        angular_acc = -2 * (phi_dot * r_dot / r)
        return [r_dot, radial_acc, phi_dot, angular_acc]

    t_eval = np.arange(0, duration, 1)
    solution = solve_ivp(rhs, [0, duration], initial_state, method='RK45', dense_output=True)
    return solution.sol(t_eval)


# Построение полной траектории

def compute_full_trajectory(initial_state):
    """Формирует полную траекторию ракеты по всем этапам полёта."""
    segments = []

    # первая ступень, атмосфера
    segments.append(
        integrate_thrust_atmosphere(
            initial_state,
            STAGE1_TIME,
            10026 * 1000,
            7983.5,
            MASS_STAGE1 + MASS_STAGE2 + MASS_STAGE3 + MASS_BREEZE1 + MASS_PAYLOAD,
            3622,
            0,
            60,
        )
    )

    # вторая ступень, атмосфера
    segments.append(
        integrate_thrust_atmosphere(
            segments[-1][:, -1],
            STAGE2_TIME,
            2400 * 1000,
            0,
            MASS_STAGE2 + MASS_STAGE3 + MASS_BREEZE1 + MASS_PAYLOAD,
            731.63,
            60,
            60,
        )
    )

    # третья ступень, вакуум
    segments.append(
        integrate_thrust_vacuum(
            segments[-1][:, -1],
            STAGE3_TIME,
            583 * 1000,
            MASS_STAGE3 + MASS_BREEZE1 + MASS_PAYLOAD,
            180,
            60,
            60,
        )
    )

    # баллистика
    segments.append(integrate_coasting(segments[-1][:, -1], COAST1_TIME))

    # первый пуск "Бриза-М"
    segments.append(
        integrate_thrust_vacuum(
            segments[-1][:, -1],
            BREEZE1_TIME,
            150 * 1000,
            MASS_BREEZE1 + MASS_PAYLOAD,
            2.57,
            60,
            80,
        )
    )

    # баллистика
    segments.append(integrate_coasting(segments[-1][:, -1], COAST2_TIME))

    # второй пуск "Бриза-М"
    segments.append(
        integrate_thrust_vacuum(
            segments[-1][:, -1],
            BREEZE2_TIME,
            32.2 * 1000,
            MASS_BREEZE2 + MASS_PAYLOAD,
            2.57,
            90,
            90,
        )
    )

    # баллистика
    segments.append(integrate_coasting(segments[-1][:, -1], COAST3_TIME))

    # третий пуск "Бриза-М"
    segments.append(
        integrate_thrust_vacuum(
            segments[-1][:, -1],
            BREEZE3_TIME,
            39.7 * 1000,
            MASS_BREEZE3 + MASS_PAYLOAD,
            2.57,
            89.6,
            89.6,
        )
    )

    # финальный участок автономного полёта
    segments.append(integrate_coasting(segments[-1][:, -1], COAST4_TIME))

    return segments


def merge_trajectory_segments(segments):
    """Объединяет все этапы в одну временную и фазовую траекторию."""
    total_time = (
        STAGE1_TIME + STAGE2_TIME + STAGE3_TIME +
        COAST1_TIME + BREEZE1_TIME + COAST2_TIME +
        BREEZE2_TIME + COAST3_TIME + BREEZE3_TIME +
        COAST4_TIME
    )

    t_global = np.arange(0, total_time, 1)
    r_all = np.concatenate([seg[0, :] for seg in segments])
    r_dot_all = np.concatenate([seg[1, :] for seg in segments])
    phi_all = np.concatenate([seg[2, :] for seg in segments])
    phi_dot_all = np.concatenate([seg[3, :] for seg in segments])

    state_all = np.array([r_all, r_dot_all, phi_all, phi_dot_all])
    return t_global, state_all


# Визуализация

def plot_altitude(ax, segments):
    """Строит график высоты от времени."""
    t, state = merge_trajectory_segments(segments)
    altitude = state[0, :] - R_PLANET
    ax.plot(t[:28000], altitude[:28000])
    ax.set_xlabel('t, с')
    ax.set_ylabel('h, м')
    ax.grid()


def plot_velocity(ax, segments):
    """Строит график скорости ракеты от времени."""
    t, state = merge_trajectory_segments(segments)
    v = np.sqrt(state[1, :]**2 + (state[0, :] * state[3, :])**2)
    ax.plot(t[:28000], v[:28000])
    ax.set_xlabel('t, с')
    ax.set_ylabel('V, м/с')
    ax.grid()


def plot_flight_profiles(segments):
    """Строит два графика: высота и скорость во времени."""
    fig, axes = plt.subplots(nrows=2, figsize=(8, 8))
    plot_altitude(axes[0], segments)
    plot_velocity(axes[1], segments)
    plt.tight_layout()
    plt.show()


# Запуск расчёта и построения графиков

full_trajectory = compute_full_trajectory([R_PLANET, 0, 0, 0])
plot_flight_profiles(full_trajectory)
