import krpc
import matplotlib.pyplot as plt
import krpc.error

conn = krpc.connect(name="Proton-M logger (manual stop)")
vessel = conn.space_center.active_vessel
body = vessel.orbit.body

time_values = []
altitude_values = []
speed_values = []

print("Сбор данных начат...")
print("Если нужно построить графики, остановите программу.")

try:
    while conn.krpc.current_game_scene == conn.krpc.GameScene.flight:
        try:
            t = conn.space_center.ut
            flight = vessel.flight()  # относительно поверхности
            altitude = flight.surface_altitude
            speed = vessel.flight(body.reference_frame).speed

        except (krpc.error.RPCError, krpc.error.ConnectionError):
            print("Связь с kRPC потеряна, выход из цикла")
            break

        time_values.append(t)
        altitude_values.append(altitude)
        speed_values.append(speed)

except KeyboardInterrupt:
    print("\nОстановка . Строим графики...")

if time_values:
    t0 = time_values[0]
    time_values = [ti - t0 for ti in time_values]

    fig, axs = plt.subplots(nrows=2, figsize=(9, 7))

    # 1) высота
    axs[0].plot(time_values, altitude_values)
    axs[0].set_xlabel("t, с")
    axs[0].set_ylabel("h, м")
    axs[0].grid(True)

    # 2) скорость
    axs[1].plot(time_values, speed_values)
    axs[1].set_xlabel("t, с")
    axs[1].set_ylabel("V, м/с")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
else:
    print("Данных нет")
