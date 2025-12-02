/**
 * Физические симуляции для изучения производных
 * Включает гармонические колебания, падение с сопротивлением, уравнение Циолковского
 */

class PhysicsSimulations {
    constructor() {
        this.simulations = {
            harmonic: null,
            falling: null,
            rocket: null
        };
        
        this.constants = {
            g: 9.81,           // ускорение свободного падения (м/с²)
            R: 8.314,          // универсальная газовая постоянная
            atm: 101325,       // атмосферное давление (Па)
            c: 299792458       // скорость света (м/с)
        };
    }

    class HarmonicOscillator {
        constructor(config = {}) {
            this.m = config.mass || 1.0;           // масса (кг)
            this.k = config.springConstant || 2.0; // жесткость (Н/м)
            this.A = config.amplitude || 1.0;      // амплитуда (м)
            this.phi = config.phase || 0;          // начальная фаза
            
            this.time = 0;
            this.isRunning = false;
            this.animationId = null;
            
            this.calculateOmega();
        }
        
        calculateOmega() {
            this.omega = Math.sqrt(this.k / this.m);
            this.period = 2 * Math.PI / this.omega;
            this.frequency = 1 / this.period;
        }
        
        updateParameters(params) {
            if (params.mass !== undefined) this.m = params.mass;
            if (params.springConstant !== undefined) this.k = params.springConstant;
            if (params.amplitude !== undefined) this.A = params.amplitude;
            if (params.phase !== undefined) this.phi = params.phase;
            
            this.calculateOmega();
        }
        
        position(t) {
            return this.A * Math.cos(this.omega * t + this.phi);
        }
        
        velocity(t) {
            return -this.A * this.omega * Math.sin(this.omega * t + this.phi);
        }
        
        acceleration(t) {
            return -this.A * this.omega * this.omega * Math.cos(this.omega * t + this.phi);
        }
        
        energy(t) {
            const x = this.position(t);
            const v = this.velocity(t);
            const kinetic = 0.5 * this.m * v * v;
            const potential = 0.5 * this.k * x * x;
            return {
                kinetic,
                potential,
                total: kinetic + potential
            };
        }
        
        startAnimation(callback, dt = 0.016) {
            this.isRunning = true;
            const animate = () => {
                if (!this.isRunning) return;
                
                this.time += dt;
                if (callback) {
                    callback({
                        time: this.time,
                        position: this.position(this.time),
                        velocity: this.velocity(this.time),
                        acceleration: this.acceleration(this.time),
                        energy: this.energy(this.time)
                    });
                }
                
                this.animationId = requestAnimationFrame(() => animate());
            };
            animate();
        }
        
        stopAnimation() {
            this.isRunning = false;
            if (this.animationId) {
                cancelAnimationFrame(this.animationId);
            }
        }
        
        // Метод Рунге-Кутты для затухающих колебаний
        solveDampedOscillator(zeta, tMax = 10, dt = 0.01) {
            // Уравнение: m*x'' + c*x' + k*x = 0
            const c = 2 * zeta * Math.sqrt(this.m * this.k); // коэффициент демпфирования
            
            const result = {
                time: [],
                position: [],
                velocity: [],
                acceleration: []
            };
            
            let x = this.A;
            let v = 0;
            
            for (let t = 0; t <= tMax; t += dt) {
                // Метод Рунге-Кутты 4-го порядка
                const k1x = v;
                const k1v = (-c * v - this.k * x) / this.m;
                
                const k2x = v + 0.5 * dt * k1v;
                const k2v = (-c * (v + 0.5 * dt * k1v) - this.k * (x + 0.5 * dt * k1x)) / this.m;
                
                const k3x = v + 0.5 * dt * k2v;
                const k3v = (-c * (v + 0.5 * dt * k2v) - this.k * (x + 0.5 * dt * k2x)) / this.m;
                
                const k4x = v + dt * k3v;
                const k4v = (-c * (v + dt * k3v) - this.k * (x + dt * k3x)) / this.m;
                
                x = x + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x);
                v = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v);
                
                result.time.push(t);
                result.position.push(x);
                result.velocity.push(v);
                result.acceleration.push((-c * v - this.k * x) / this.m);
            }
            
            return result;
        }
    }

    /**
     * Падение тела с сопротивлением воздуха
     */
    class FallingBody {
        constructor(config = {}) {
            this.m = config.mass || 70;        // масса (кг)
            this.area = config.area || 0.5;    // площадь поперечного сечения (м²)
            this.Cd = config.dragCoefficient || 0.47; // коэффициент лобового сопротивления
            this.rho = config.airDensity || 1.225; // плотность воздуха (кг/м³)
            this.h0 = config.initialHeight || 1000; // начальная высота (м)
            
            this.g = 9.81;
            this.time = 0;
            this.isRunning = false;
        }
        
        // Сила сопротивления: Fd = 0.5 * ρ * v² * Cd * A
        dragForce(velocity) {
            return 0.5 * this.rho * velocity * Math.abs(velocity) * this.Cd * this.area;
        }
        
        // Уравнение движения: m*dv/dt = mg - Fd
        solveEuler(dt = 0.01, maxTime = 100) {
            const result = {
                time: [],
                height: [],
                velocity: [],
                acceleration: [],
                dragForce: []
            };
            
            let h = this.h0;
            let v = 0;
            let t = 0;
            
            while (h > 0 && t < maxTime) {
                const Fd = this.dragForce(v);
                const a = this.g - Fd / this.m;
                
                // Метод Эйлера
                v += a * dt;
                h -= v * dt;
                t += dt;
                
                // Проверка на достижение земли
                if (h < 0) {
                    h = 0;
                    // Линейная интерполяция для точного времени удара
                    const dtImpact = -h / v;
                    t += dtImpact;
                }
                
                result.time.push(t);
                result.height.push(h);
                result.velocity.push(v);
                result.acceleration.push(a);
                result.dragForce.push(Fd);
            }
            
            return result;
        }
        
        // Аналитическое решение для линейного сопротивления (Fd = -bv)
        solveLinearDrag(b, dt = 0.01, maxTime = 100) {
            const result = {
                time: [],
                height: [],
                velocity: [],
                terminalVelocity: this.m * this.g / b
            };
            
            for (let t = 0; t <= maxTime; t += dt) {
                const v = result.terminalVelocity * (1 - Math.exp(-b * t / this.m));
                const h = this.h0 - result.terminalVelocity * t + 
                         (this.m * result.terminalVelocity / b) * (1 - Math.exp(-b * t / this.m));
                
                result.time.push(t);
                result.height.push(Math.max(h, 0));
                result.velocity.push(v);
            }
            
            return result;
        }
        
        // Скорость терминального падения
        terminalVelocity() {
            // Для квадратичного сопротивления: v_term = sqrt(2mg / (ρACd))
            return Math.sqrt((2 * this.m * this.g) / (this.rho * this.area * this.Cd));
        }
        
        // Время падения (приблизительное)
        timeToImpact(method = 'euler') {
            const solution = method === 'euler' ? 
                this.solveEuler(0.01, 1000) : 
                this.solveLinearDrag(0.5 * this.rho * this.area * this.Cd, 0.01, 1000);
            
            return solution.time[solution.time.length - 1];
        }
    }

    /**
     * Реактивное движение (уравнение Циолковского)
     */
    class RocketMotion {
        constructor(config = {}) {
            this.m0 = config.initialMass || 1000;      // начальная масса (кг)
            this.mf = config.finalMass || 100;         // конечная масса (кг)
            this.ve = config.exhaustVelocity || 3000;  // скорость истечения (м/с)
            this.dmdt = config.fuelRate || 10;         // расход топлива (кг/с)
            
            this.time = 0;
            this.isRunning = false;
        }
        
        // Уравнение Циолковского
        deltaV() {
            return this.ve * Math.log(this.m0 / this.mf);
        }
        
        // Масса как функция времени
        mass(t) {
            const burnt = this.dmdt * t;
            return Math.max(this.m0 - burnt, this.mf);
        }
        
        // Скорость как функция времени (упрощенное решение)
        velocity(t) {
            const m = this.mass(t);
            return this.ve * Math.log(this.m0 / m);
        }
        
        // Ускорение как функция времени
        acceleration(t) {
            // a = ve * (dm/dt) / m(t)
            const m = this.mass(t);
            if (m > this.mf) {
                return this.ve * this.dmdt / m;
            }
            return 0;
        }
        
        // Решение уравнения движения
        solve(dt = 0.1, maxTime = 100) {
            const result = {
                time: [],
                mass: [],
                velocity: [],
                acceleration: [],
                thrust: [],
                deltaV: []
            };
            
            let m = this.m0;
            let v = 0;
            let t = 0;
            
            while (m > this.mf && t < maxTime) {
                // Сила тяги
                const thrust = this.ve * this.dmdt;
                
                // Ускорение (пренебрегаем гравитацией и сопротивлением)
                const a = thrust / m;
                
                // Интегрирование
                v += a * dt;
                m -= this.dmdt * dt;
                t += dt;
                
                result.time.push(t);
                result.mass.push(m);
                result.velocity.push(v);
                result.acceleration.push(a);
                result.thrust.push(thrust);
                result.deltaV.push(this.ve * Math.log(this.m0 / m));
            }
            
            return result;
        }
        
        // Многоступенчатая ракета
        multiStageDeltaV(stages) {
            let totalDeltaV = 0;
            let currentMass = this.m0;
            
            stages.forEach((stage, index) => {
                const stageMass = stage.mass;
                const stageVe = stage.exhaustVelocity || this.ve;
                const stageMassRatio = currentMass / (currentMass - stageMass);
                totalDeltaV += stageVe * Math.log(stageMassRatio);
                currentMass -= stageMass;
            });
            
            return totalDeltaV;
        }
        
        // Требуемая масса для заданной Δv
        requiredMassRatio(deltaV) {
            return Math.exp(deltaV / this.ve);
        }
    }

    /**
     * Электромагнитные колебания (RLC-цепь)
     */
    class RLCCircuit {
        constructor(config = {}) {
            this.R = config.resistance || 100;     // сопротивление (Ом)
            this.L = config.inductance || 0.1;     // индуктивность (Гн)
            this.C = config.capacitance || 1e-6;   // емкость (Ф)
            this.V0 = config.initialVoltage || 10; // начальное напряжение (В)
            
            this.calculateParameters();
        }
        
        calculateParameters() {
            // Собственная частота
            this.omega0 = 1 / Math.sqrt(this.L * this.C);
            
            // Коэффициент затухания
            this.alpha = this.R / (2 * this.L);
            
            // Добротность
            this.Q = 1 / this.R * Math.sqrt(this.L / this.C);
            
            // Режим колебаний
            if (this.alpha < this.omega0) {
                this.mode = 'underdamped';      // колебательный
                this.omega = Math.sqrt(this.omega0 * this.omega0 - this.alpha * this.alpha);
            } else if (this.alpha > this.omega0) {
                this.mode = 'overdamped';       // апериодический
            } else {
                this.mode = 'criticallyDamped'; // критический
            }
        }
        
        // Напряжение на конденсаторе
        voltage(t) {
            if (this.mode === 'underdamped') {
                return this.V0 * Math.exp(-this.alpha * t) * 
                       (Math.cos(this.omega * t) + (this.alpha / this.omega) * Math.sin(this.omega * t));
            } else if (this.mode === 'overdamped') {
                const s1 = -this.alpha + Math.sqrt(this.alpha * this.alpha - this.omega0 * this.omega0);
                const s2 = -this.alpha - Math.sqrt(this.alpha * this.alpha - this.omega0 * this.omega0);
                const A = (s2 * this.V0) / (s2 - s1);
                const B = (-s1 * this.V0) / (s2 - s1);
                return A * Math.exp(s1 * t) + B * Math.exp(s2 * t);
            } else {
                return this.V0 * (1 + this.alpha * t) * Math.exp(-this.alpha * t);
            }
        }
        
        // Ток в цепи
        current(t) {
            // I = C * dV/dt
            const h = 0.0001;
            const v1 = this.voltage(t);
            const v2 = this.voltage(t + h);
            return -this.C * (v2 - v1) / h;
        }
        
        solve(dt = 0.0001, maxTime = 0.01) {
            const result = {
                time: [],
                voltage: [],
                current: [],
                energy: []
            };
            
            for (let t = 0; t <= maxTime; t += dt) {
                const v = this.voltage(t);
                const i = this.current(t);
                const energy = 0.5 * this.L * i * i + 0.5 * this.C * v * v;
                
                result.time.push(t);
                result.voltage.push(v);
                result.current.push(i);
                result.energy.push(energy);
            }
            
            return result;
        }
    }

    /**
     * Теплопроводность (уравнение теплопроводности)
     */
    class HeatConduction {
        constructor(config = {}) {
            this.k = config.thermalConductivity || 0.5;  // коэффициент теплопроводности
            this.rho = config.density || 1000;           // плотность
            this.c = config.specificHeat || 4186;        // удельная теплоемкость
            this.L = config.length || 1.0;               // длина стержня (м)
            
            this.alpha = this.k / (this.rho * this.c);   // коэффициент температуропроводности
        }
        
        // Аналитическое решение для бесконечного стержня
        infiniteRodSolution(x, t, T0 = 100, Tinf = 0) {
            if (t <= 0) return T0;
            
            const eta = x / Math.sqrt(4 * this.alpha * t);
            // Интеграл вероятности
            const erf = (z) => {
                const t = 1.0 / (1.0 + 0.5 * Math.abs(z));
                const ans = 1 - t * Math.exp(-z*z - 1.26551223 +
                    t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
                    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
                    t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
                return z >= 0 ? ans : -ans;
            };
            
            return Tinf + (T0 - Tinf) * erf(eta);
        }
        
        // Численное решение методом конечных разностей
        solveFiniteDifference(Nx = 100, Nt = 1000, T0 = 100, Tends = 0) {
            const dx = this.L / (Nx - 1);
            const dt = 0.25 * dx * dx / this.alpha; // условие устойчивости
            
            // Начальное распределение температуры
            let T = new Array(Nx);
            for (let i = 0; i < Nx; i++) {
                T[i] = T0;
            }
            
            // Граничные условия
            T[0] = Tends;
            T[Nx - 1] = Tends;
            
            const result = {
                time: [0],
                temperature: [T.slice()],
                x: Array.from({length: Nx}, (_, i) => i * dx)
            };
            
            // Временные шаги
            for (let n = 1; n <= Nt; n++) {
                const Tnew = new Array(Nx);
                Tnew[0] = Tends;
                Tnew[Nx - 1] = Tends;
                
                // Явная схема
                for (let i = 1; i < Nx - 1; i++) {
                    Tnew[i] = T[i] + this.alpha * dt / (dx * dx) * 
                             (T[i+1] - 2*T[i] + T[i-1]);
                }
                
                T = Tnew;
                
                if (n % 10 === 0) {
                    result.time.push(n * dt);
                    result.temperature.push(T.slice());
                }
            }
            
            return result;
        }
        
        // Градиент температуры
        temperatureGradient(x, t, T0 = 100, Tinf = 0) {
            const h = 0.001;
            const T1 = this.infiniteRodSolution(x - h, t, T0, Tinf);
            const T2 = this.infiniteRodSolution(x + h, t, T0, Tinf);
            return (T2 - T1) / (2 * h);
        }
        
        // Тепловой поток (закон Фурье)
        heatFlux(x, t, T0 = 100, Tinf = 0) {
            const gradT = this.temperatureGradient(x, t, T0, Tinf);
            return -this.k * gradT;
        }
    }

    // Экспорт классов
    return {
        HarmonicOscillator,
        FallingBody,
        RocketMotion,
        RLCCircuit,
        HeatConduction,
        
        // Вспомогательные функции
        calculateDerivative: function(f, x, h = 0.0001) {
            return (f(x + h) - f(x - h)) / (2 * h);
        },
        
        calculateSecondDerivative: function(f, x, h = 0.0001) {
            return (f(x + h) - 2*f(x) + f(x - h)) / (h * h);
        },
        
        // Примеры физических законов через производные
        examples: {
            // Законы Ньютона
            newtonsSecondLaw: function(m, a) {
                return m * a;
            },
            
            // Закон Гука
            hookesLaw: function(k, x) {
                return -k * x;
            },
            
            // Закон Ома
            ohmsLaw: function(V, R) {
                return V / R;
            },
            
            // Уравнение состояния идеального газа
            idealGasLaw: function(P, V, n, T) {
                const R = 8.314;
                return (P * V) - (n * R * T);
            },
            
            // Уравнение Эйнштейна
            einsteinEnergy: function(m) {
                return m * 299792458 * 299792458;
            }
        }
    };
}

// Создаем глобальный экземпляр
if (typeof window !== 'undefined') {
    window.PhysicsSimulations = new PhysicsSimulations();
}

export default PhysicsSimulations;