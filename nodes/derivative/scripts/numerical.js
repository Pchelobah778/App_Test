/**
 * Численные методы дифференцирования и решения дифференциальных уравнений
 * Методы конечных разностей, Рунге-Кутты, автоматическое дифференцирование
 */

class NumericalMethods {
    constructor() {
        this.methods = {
            differentiation: {},
            integration: {},
            ode: {},
            optimization: {}
        };
        
        this.precision = 1e-12;
        this.maxIterations = 1000;
    }

    /**
     * Методы численного дифференцирования
     */
    class NumericalDifferentiation {
        constructor() {
            this.methods = {
                forward: this.forwardDifference.bind(this),
                backward: this.backwardDifference.bind(this),
                central: this.centralDifference.bind(this),
                fivePoint: this.fivePointDifference.bind(this),
                richardson: this.richardsonExtrapolation.bind(this),
                automatic: this.automaticDifferentiation.bind(this)
            };
        }
        
        // Правая разность
        forwardDifference(f, x, h = 0.001) {
            return (f(x + h) - f(x)) / h;
        }
        
        // Левая разность
        backwardDifference(f, x, h = 0.001) {
            return (f(x) - f(x - h)) / h;
        }
        
        // Центральная разность
        centralDifference(f, x, h = 0.001) {
            return (f(x + h) - f(x - h)) / (2 * h);
        }
        
        // Пять точек (4-го порядка точности)
        fivePointDifference(f, x, h = 0.001) {
            return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12 * h);
        }
        
        // Экстраполяция Ричардсона
        richardsonExtrapolation(f, x, method = 'central', h0 = 0.1, steps = 5) {
            const D = new Array(steps);
            const h = new Array(steps);
            
            // Первый столбец - различные шаги
            for (let i = 0; i < steps; i++) {
                h[i] = h0 / Math.pow(2, i);
                switch(method) {
                    case 'forward':
                        D[i] = [this.forwardDifference(f, x, h[i])];
                        break;
                    case 'backward':
                        D[i] = [this.backwardDifference(f, x, h[i])];
                        break;
                    case 'central':
                        D[i] = [this.centralDifference(f, x, h[i])];
                        break;
                }
            }
            
            // Экстраполяция
            for (let j = 1; j < steps; j++) {
                for (let i = j; i < steps; i++) {
                    D[i][j] = D[i][j-1] + (D[i][j-1] - D[i-1][j-1]) / 
                             (Math.pow(2, j * 2) - 1);
                }
            }
            
            return {
                approximations: D,
                bestEstimate: D[steps-1][steps-1],
                errorEstimate: Math.abs(D[steps-1][steps-1] - D[steps-2][steps-2])
            };
        }
        
        // Автоматическое дифференцирование (режим прямой передачи)
        automaticDifferentiation(f, x) {
            // Простая реализация через дуальные числа
            class Dual {
                constructor(real, epsilon = 0) {
                    this.real = real;
                    this.epsilon = epsilon; // производная
                }
                
                add(other) {
                    return new Dual(
                        this.real + other.real,
                        this.epsilon + other.epsilon
                    );
                }
                
                sub(other) {
                    return new Dual(
                        this.real - other.real,
                        this.epsilon - other.epsilon
                    );
                }
                
                mul(other) {
                    return new Dual(
                        this.real * other.real,
                        this.real * other.epsilon + this.epsilon * other.real
                    );
                }
                
                div(other) {
                    return new Dual(
                        this.real / other.real,
                        (this.epsilon * other.real - this.real * other.epsilon) / (other.real * other.real)
                    );
                }
                
                pow(n) {
                    return new Dual(
                        Math.pow(this.real, n),
                        n * Math.pow(this.real, n - 1) * this.epsilon
                    );
                }
                
                sin() {
                    return new Dual(
                        Math.sin(this.real),
                        Math.cos(this.real) * this.epsilon
                    );
                }
                
                cos() {
                    return new Dual(
                        Math.cos(this.real),
                        -Math.sin(this.real) * this.epsilon
                    );
                }
                
                exp() {
                    const e = Math.exp(this.real);
                    return new Dual(e, e * this.epsilon);
                }
                
                log() {
                    return new Dual(
                        Math.log(this.real),
                        this.epsilon / this.real
                    );
                }
            }
            
            // Конвертируем функцию для работы с дуальными числами
            const dualX = new Dual(x, 1); // производная по x = 1
            const result = f(dualX);
            
            return {
                value: result.real,
                derivative: result.epsilon
            };
        }
        
        // Вторая производная
        secondDerivative(f, x, h = 0.001) {
            return (f(x + h) - 2*f(x) + f(x - h)) / (h * h);
        }
        
        // Частные производные для многомерных функций
        partialDerivative(f, point, varIndex, h = 0.001) {
            const pointPlus = [...point];
            const pointMinus = [...point];
            
            pointPlus[varIndex] += h;
            pointMinus[varIndex] -= h;
            
            return (f(...pointPlus) - f(...pointMinus)) / (2 * h);
        }
        
        // Градиент численный
        gradient(f, point, h = 0.001) {
            const grad = [];
            for (let i = 0; i < point.length; i++) {
                grad.push(this.partialDerivative(f, point, i, h));
            }
            return grad;
        }
        
        // Гессиан (матрица вторых производных)
        hessian(f, point, h = 0.001) {
            const n = point.length;
            const H = Array(n).fill().map(() => Array(n).fill(0));
            
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    if (i === j) {
                        // Диагональные элементы
                        const pointPlus = [...point];
                        const pointMinus = [...point];
                        pointPlus[i] += h;
                        pointMinus[i] -= h;
                        H[i][i] = (f(...pointPlus) - 2*f(...point) + f(...pointMinus)) / (h * h);
                    } else {
                        // Смешанные производные
                        const point1 = [...point];
                        const point2 = [...point];
                        const point3 = [...point];
                        const point4 = [...point];
                        
                        point1[i] += h; point1[j] += h;
                        point2[i] += h; point2[j] -= h;
                        point3[i] -= h; point3[j] += h;
                        point4[i] -= h; point4[j] -= h;
                        
                        H[i][j] = (f(...point1) - f(...point2) - f(...point3) + f(...point4)) / (4 * h * h);
                    }
                }
            }
            
            return H;
        }
        
        // Анализ ошибок
        errorAnalysis(f, x, exactDerivative, hValues = [1, 0.5, 0.25, 0.125, 0.0625]) {
            const results = [];
            
            for (const h of hValues) {
                const forward = this.forwardDifference(f, x, h);
                const backward = this.backwardDifference(f, x, h);
                const central = this.centralDifference(f, x, h);
                const fivePoint = this.fivePointDifference(f, x, h);
                
                results.push({
                    h,
                    forward: {
                        value: forward,
                        error: Math.abs(forward - exactDerivative),
                        relativeError: Math.abs(forward - exactDerivative) / Math.abs(exactDerivative)
                    },
                    backward: {
                        value: backward,
                        error: Math.abs(backward - exactDerivative),
                        relativeError: Math.abs(backward - exactDerivative) / Math.abs(exactDerivative)
                    },
                    central: {
                        value: central,
                        error: Math.abs(central - exactDerivative),
                        relativeError: Math.abs(central - exactDerivative) / Math.abs(exactDerivative)
                    },
                    fivePoint: {
                        value: fivePoint,
                        error: Math.abs(fivePoint - exactDerivative),
                        relativeError: Math.abs(fivePoint - exactDerivative) / Math.abs(exactDerivative)
                    }
                });
            }
            
            return results;
        }
        
        // Порядок точности метода
        orderOfAccuracy(f, x, method, exactDerivative) {
            const h1 = 0.1;
            const h2 = 0.05;
            
            const D1 = method(f, x, h1);
            const D2 = method(f, x, h2);
            
            const error1 = Math.abs(D1 - exactDerivative);
            const error2 = Math.abs(D2 - exactDerivative);
            
            return Math.log(error1 / error2) / Math.log(h1 / h2);
        }
    }

    /**
     * Методы численного интегрирования (для сравнения)
     */
    class NumericalIntegration {
        constructor() {
            this.methods = {
                rectangle: this.rectangleMethod.bind(this),
                trapezoidal: this.trapezoidalMethod.bind(this),
                simpson: this.simpsonMethod.bind(this),
                romberg: this.rombergIntegration.bind(this),
                monteCarlo: this.monteCarloIntegration.bind(this),
                gaussian: this.gaussianQuadrature.bind(this)
            };
        }
        
        // Метод прямоугольников
        rectangleMethod(f, a, b, n = 100) {
            const dx = (b - a) / n;
            let sum = 0;
            
            for (let i = 0; i < n; i++) {
                const x = a + (i + 0.5) * dx; // средняя точка
                sum += f(x) * dx;
            }
            
            return sum;
        }
        
        // Метод трапеций
        trapezoidalMethod(f, a, b, n = 100) {
            const dx = (b - a) / n;
            let sum = 0.5 * (f(a) + f(b));
            
            for (let i = 1; i < n; i++) {
                const x = a + i * dx;
                sum += f(x);
            }
            
            return sum * dx;
        }
        
        // Метод Симпсона
        simpsonMethod(f, a, b, n = 100) {
            if (n % 2 !== 0) n++; // делаем четным
            
            const dx = (b - a) / n;
            let sum = f(a) + f(b);
            
            for (let i = 1; i < n; i++) {
                const x = a + i * dx;
                if (i % 2 === 0) {
                    sum += 2 * f(x);
                } else {
                    sum += 4 * f(x);
                }
            }
            
            return sum * dx / 3;
        }
        
        // Метод Ромберга
        rombergIntegration(f, a, b, maxSteps = 10) {
            const R = Array(maxSteps).fill().map(() => Array(maxSteps).fill(0));
            
            // Первый столбец - составная трапеция
            for (let i = 0; i < maxSteps; i++) {
                const n = Math.pow(2, i);
                R[i][0] = this.trapezoidalMethod(f, a, b, n);
            }
            
            // Экстраполяция Ричардсона
            for (let j = 1; j < maxSteps; j++) {
                for (let i = j; i < maxSteps; i++) {
                    R[i][j] = R[i][j-1] + (R[i][j-1] - R[i-1][j-1]) / 
                             (Math.pow(4, j) - 1);
                }
            }
            
            return {
                approximations: R,
                bestEstimate: R[maxSteps-1][maxSteps-1],
                errorEstimate: Math.abs(R[maxSteps-1][maxSteps-1] - R[maxSteps-2][maxSteps-2])
            };
        }
        
        // Метод Монте-Карло
        monteCarloIntegration(f, a, b, n = 10000) {
            let sum = 0;
            let sumSq = 0;
            
            for (let i = 0; i < n; i++) {
                const x = a + Math.random() * (b - a);
                const fx = f(x);
                sum += fx;
                sumSq += fx * fx;
            }
            
            const mean = sum / n;
            const variance = (sumSq / n - mean * mean) / n;
            
            return {
                estimate: mean * (b - a),
                standardError: Math.sqrt(variance) * (b - a),
                samples: n
            };
        }
        
        // Квадратура Гаусса
        gaussianQuadrature(f, a, b, n = 5) {
            // Узлы и веса для разных n
            const nodesWeights = {
                2: {
                    nodes: [-0.5773502691896257, 0.5773502691896257],
                    weights: [1, 1]
                },
                3: {
                    nodes: [-0.7745966692414834, 0, 0.7745966692414834],
                    weights: [0.5555555555555556, 0.8888888888888888, 0.5555555555555556]
                },
                5: {
                    nodes: [
                        -0.9061798459386640,
                        -0.5384693101056831,
                        0,
                        0.5384693101056831,
                        0.9061798459386640
                    ],
                    weights: [
                        0.2369268850561891,
                        0.4786286704993665,
                        0.5688888888888889,
                        0.4786286704993665,
                        0.2369268850561891
                    ]
                }
            };
            
            const { nodes, weights } = nodesWeights[n] || nodesWeights[5];
            
            // Преобразование из [-1, 1] к [a, b]
            const transform = (t) => (b - a) / 2 * t + (a + b) / 2;
            const jacobian = (b - a) / 2;
            
            let sum = 0;
            for (let i = 0; i < n; i++) {
                const x = transform(nodes[i]);
                sum += weights[i] * f(x);
            }
            
            return sum * jacobian;
        }
        
        // Несобственные интегралы
        improperIntegral(f, a, Infinity, method = 'simpson', subdivisions = 10) {
            // Преобразование t = 1/x
            const g = (t) => {
                if (t === 0) return 0;
                const x = 1 / t;
                return f(x) / (t * t);
            };
            
            return this[method](g, 0, 1/a, subdivisions);
        }
        
        // Кратные интегралы
        doubleIntegral(f, ax, bx, ay, by, nx = 50, ny = 50) {
            const dx = (bx - ax) / nx;
            const dy = (by - ay) / ny;
            let sum = 0;
            
            for (let i = 0; i < nx; i++) {
                const x = ax + (i + 0.5) * dx;
                for (let j = 0; j < ny; j++) {
                    const y = ay + (j + 0.5) * dy;
                    sum += f(x, y);
                }
            }
            
            return sum * dx * dy;
        }
    }

    /**
     * Решение обыкновенных дифференциальных уравнений
     */
    class ODESolver {
        constructor() {
            this.methods = {
                euler: this.eulerMethod.bind(this),
                heun: this.heunMethod.bind(this),
                midpoint: this.midpointMethod.bind(this),
                rk4: this.rungeKutta4.bind(this),
                adaptiveRK: this.adaptiveRungeKutta.bind(this),
                verlet: this.verletIntegration.bind(this)
            };
        }
        
        // Метод Эйлера
        eulerMethod(f, y0, t0, tf, n = 100) {
            const dt = (tf - t0) / n;
            const t = [t0];
            const y = [y0];
            
            for (let i = 0; i < n; i++) {
                const yi = y[i];
                const ti = t[i];
                
                const k1 = f(ti, yi);
                const yNext = yi + dt * k1;
                
                t.push(ti + dt);
                y.push(yNext);
            }
            
            return { t, y };
        }
        
        // Метод Хойна (улучшенный Эйлер)
        heunMethod(f, y0, t0, tf, n = 100) {
            const dt = (tf - t0) / n;
            const t = [t0];
            const y = [y0];
            
            for (let i = 0; i < n; i++) {
                const yi = y[i];
                const ti = t[i];
                
                const k1 = f(ti, yi);
                const k2 = f(ti + dt, yi + dt * k1);
                
                const yNext = yi + dt * 0.5 * (k1 + k2);
                
                t.push(ti + dt);
                y.push(yNext);
            }
            
            return { t, y };
        }
        
        // Метод средней точки
        midpointMethod(f, y0, t0, tf, n = 100) {
            const dt = (tf - t0) / n;
            const t = [t0];
            const y = [y0];
            
            for (let i = 0; i < n; i++) {
                const yi = y[i];
                const ti = t[i];
                
                const k1 = f(ti, yi);
                const k2 = f(ti + 0.5 * dt, yi + 0.5 * dt * k1);
                
                const yNext = yi + dt * k2;
                
                t.push(ti + dt);
                y.push(yNext);
            }
            
            return { t, y };
        }
        
        // Метод Рунге-Кутты 4-го порядка
        rungeKutta4(f, y0, t0, tf, n = 100) {
            const dt = (tf - t0) / n;
            const t = [t0];
            const y = [y0];
            
            for (let i = 0; i < n; i++) {
                const yi = y[i];
                const ti = t[i];
                
                const k1 = f(ti, yi);
                const k2 = f(ti + 0.5 * dt, yi + 0.5 * dt * k1);
                const k3 = f(ti + 0.5 * dt, yi + 0.5 * dt * k2);
                const k4 = f(ti + dt, yi + dt * k3);
                
                const yNext = yi + dt * (k1 + 2*k2 + 2*k3 + k4) / 6;
                
                t.push(ti + dt);
                y.push(yNext);
            }
            
            return { t, y };
        }
        
        // Адаптивный метод Рунге-Кутты (метод Фельберга 4-5)
        adaptiveRungeKutta(f, y0, t0, tf, tol = 1e-6, h0 = 0.1) {
            const t = [t0];
            const y = [y0];
            const h = [h0];
            
            let ti = t0;
            let yi = y0;
            let hi = h0;
            
            while (ti < tf) {
                // Коэффициенты Фельберга
                const k1 = f(ti, yi);
                const k2 = f(ti + hi/4, yi + hi*k1/4);
                const k3 = f(ti + 3*hi/8, yi + hi*(3*k1 + 9*k2)/32);
                const k4 = f(ti + 12*hi/13, yi + hi*(1932*k1 - 7200*k2 + 7296*k3)/2197);
                const k5 = f(ti + hi, yi + hi*(439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104));
                const k6 = f(ti + hi/2, yi + hi*(-8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40));
                
                // Решения 4-го и 5-го порядка
                const y4 = yi + hi*(25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5);
                const y5 = yi + hi*(16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55);
                
                // Оценка ошибки
                const error = Math.abs(y5 - y4);
                
                if (error <= tol) {
                    // Шаг принят
                    ti += hi;
                    yi = y5;
                    
                    t.push(ti);
                    y.push(yi);
                    h.push(hi);
                    
                    // Увеличиваем шаг
                    hi *= Math.min(2, 0.9 * Math.pow(tol/error, 0.2));
                } else {
                    // Уменьшаем шаг
                    hi *= Math.max(0.1, 0.9 * Math.pow(tol/error, 0.25));
                }
                
                // Ограничиваем шаг
                hi = Math.min(hi, tf - ti);
                
                if (hi < 1e-12) {
                    console.warn('Слишком маленький шаг');
                    break;
                }
            }
            
            return { t, y, h };
        }
        
        // Интегрирование Верле (для уравнений второго порядка)
        verletIntegration(f, x0, v0, t0, tf, n = 100) {
            const dt = (tf - t0) / n;
            const t = [t0];
            const x = [x0];
            const v = [v0];
            const a = [f(t0, x0, v0)];
            
            // Первый шаг - метод Эйлера
            let xi = x0 + v0 * dt + 0.5 * a[0] * dt * dt;
            let vi = v0 + a[0] * dt;
            let ti = t0 + dt;
            
            t.push(ti);
            x.push(xi);
            v.push(vi);
            a.push(f(ti, xi, vi));
            
            // Последующие шаги - метод Верле
            for (let i = 2; i <= n; i++) {
                const ai = f(ti, xi, vi);
                const xNext = 2*xi - x[i-2] + ai * dt * dt;
                const vNext = (xNext - x[i-2]) / (2*dt);
                
                ti += dt;
                xi = xNext;
                vi = vNext;
                
                t.push(ti);
                x.push(xi);
                v.push(vi);
                a.push(ai);
            }
            
            return { t, x, v, a };
        }
        
        // Системы ОДУ
        solveSystem(F, Y0, t0, tf, n = 100, method = 'rk4') {
            const dt = (tf - t0) / n;
            const t = [t0];
            const Y = [Y0];
            
            const solver = this.methods[method];
            
            for (let i = 0; i < n; i++) {
                const Yi = Y[i];
                const ti = t[i];
                
                let Ynext;
                
                if (method === 'rk4') {
                    const k1 = F(ti, Yi);
                    const k2 = F(ti + 0.5*dt, Yi.map((y, idx) => y + 0.5*dt*k1[idx]));
                    const k3 = F(ti + 0.5*dt, Yi.map((y, idx) => y + 0.5*dt*k2[idx]));
                    const k4 = F(ti + dt, Yi.map((y, idx) => y + dt*k3[idx]));
                    
                    Ynext = Yi.map((y, idx) => 
                        y + dt*(k1[idx] + 2*k2[idx] + 2*k3[idx] + k4[idx])/6
                    );
                } else {
                    // Для других методов
                    const k1 = F(ti, Yi);
                    Ynext = Yi.map((y, idx) => y + dt * k1[idx]);
                }
                
                t.push(ti + dt);
                Y.push(Ynext);
            }
            
            return { t, Y };
        }
        
        // Метод стрельбы для краевых задач
        shootingMethod(F, ya, yb, a, b, guess1, guess2, tol = 1e-6) {
            // Решаем с первым предположением
            const sol1 = this.rungeKutta4(F, [ya, guess1], a, b, 100);
            const error1 = sol1.y[sol1.y.length - 1][0] - yb;
            
            // Решаем со вторым предположением
            const sol2 = this.rungeKutta4(F, [ya, guess2], a, b, 100);
            const error2 = sol2.y[sol2.y.length - 1][0] - yb;
            
            // Метод секущих для нахождения корня
            let guess = guess2;
            let prevGuess = guess1;
            let prevError = error1;
            let currentError = error2;
            
            for (let iter = 0; iter < 100; iter++) {
                if (Math.abs(currentError) < tol) break;
                
                const nextGuess = guess - currentError * 
                    (guess - prevGuess) / (currentError - prevError);
                
                const sol = this.rungeKutta4(F, [ya, nextGuess], a, b, 100);
                const nextError = sol.y[sol.y.length - 1][0] - yb;
                
                prevGuess = guess;
                prevError = currentError;
                guess = nextGuess;
                currentError = nextError;
            }
            
            // Финальное решение
            return this.rungeKutta4(F, [ya, guess], a, b, 100);
        }
    }

    /**
     * Методы оптимизации (с использованием производных)
     */
    class OptimizationMethods {
        constructor() {
            this.methods = {
                gradientDescent: this.gradientDescent.bind(this),
                newton: this.newtonMethod.bind(this),
                conjugateGradient: this.conjugateGradient.bind(this),
                bfgs: this.bfgsMethod.bind(this)
            };
        }
        
        // Градиентный спуск
        gradientDescent(f, gradF, x0, alpha = 0.1, maxIter = 1000, tol = 1e-6) {
            const history = [x0];
            let x = x0;
            let fx = f(x);
            
            for (let iter = 0; iter < maxIter; iter++) {
                const gradient = gradF(x);
                const norm = Math.sqrt(gradient.reduce((sum, g) => sum + g*g, 0));
                
                if (norm < tol) break;
                
                // Шаг градиентного спуска
                x = x.map((xi, i) => xi - alpha * gradient[i]);
                
                const fxNew = f(x);
                
                // Адаптивный шаг
                if (fxNew < fx) {
                    alpha *= 1.1;
                } else {
                    alpha *= 0.5;
                }
                
                history.push([...x]);
                fx = fxNew;
            }
            
            return {
                solution: x,
                value: f(x),
                iterations: history.length,
                history: history
            };
        }
        
        // Метод Ньютона
        newtonMethod(f, gradF, hessF, x0, maxIter = 100, tol = 1e-6) {
            const history = [x0];
            let x = x0;
            
            for (let iter = 0; iter < maxIter; iter++) {
                const gradient = gradF(x);
                const hessian = hessF(x);
                const norm = Math.sqrt(gradient.reduce((sum, g) => sum + g*g, 0));
                
                if (norm < tol) break;
                
                // Решаем линейную систему HΔx = -∇f
                const deltaX = this.solveLinearSystem(hessian, 
                    gradient.map(g => -g));
                
                // Обновляем решение
                x = x.map((xi, i) => xi + deltaX[i]);
                history.push([...x]);
            }
            
            return {
                solution: x,
                value: f(x),
                iterations: history.length,
                history: history
            };
        }
        
        // Метод сопряженных градиентов
        conjugateGradient(f, gradF, x0, maxIter = 1000, tol = 1e-6) {
            const history = [x0];
            let x = x0;
            let r = gradF(x).map(g => -g); // невязка
            let p = [...r]; // направление
            let rsold = r.reduce((sum, ri) => sum + ri*ri, 0);
            
            for (let iter = 0; iter < maxIter; iter++) {
                // Линейный поиск (здесь упрощенно)
                const Ap = this.approximateHessianVector(f, x, p);
                const alpha = rsold / p.reduce((sum, pi, i) => sum + pi * Ap[i], 0);
                
                // Обновляем решение
                x = x.map((xi, i) => xi + alpha * p[i]);
                
                // Обновляем невязку
                const rNew = gradF(x).map(g => -g);
                const rsnew = rNew.reduce((sum, ri) => sum + ri*ri, 0);
                
                if (Math.sqrt(rsnew) < tol) break;
                
                // Обновляем направление
                const beta = rsnew / rsold;
                p = rNew.map((ri, i) => ri + beta * p[i]);
                
                r = rNew;
                rsold = rsnew;
                history.push([...x]);
            }
            
            return {
                solution: x,
                value: f(x),
                iterations: history.length,
                history: history
            };
        }
        
        // Метод BFGS (квазиньютоновский)
        bfgsMethod(f, gradF, x0, maxIter = 1000, tol = 1e-6) {
            const n = x0.length;
            let x = x0;
            let B = this.identityMatrix(n); // начальное приближение обратного гессиана
            let gradient = gradF(x);
            const history = [x0];
            
            for (let iter = 0; iter < maxIter; iter++) {
                const norm = Math.sqrt(gradient.reduce((sum, g) => sum + g*g, 0));
                if (norm < tol) break;
                
                // Направление поиска
                const p = this.matrixVectorMultiply(B, gradient.map(g => -g));
                
                // Линейный поиск (упрощенно)
                const alpha = 0.1; // в реальности нужно делать линейный поиск
                const xNew = x.map((xi, i) => xi + alpha * p[i]);
                const gradientNew = gradF(xNew);
                
                // Обновление BFGS
                const s = xNew.map((xNewi, i) => xNewi - x[i]);
                const y = gradientNew.map((gNew, i) => gNew - gradient[i]);
                
                const rho = 1 / s.reduce((sum, si, i) => sum + si * y[i], 0);
                
                // Обновление B
                const Bs = this.matrixVectorMultiply(B, s);
                const sBs = s.reduce((sum, si, i) => sum + si * Bs[i], 0);
                
                const term1 = s.map((si, i) => sBs * rho * rho * y[i] - rho * Bs[i]);
                const term2 = y.map((yi, i) => -rho * Bs[i]);
                
                for (let i = 0; i < n; i++) {
                    for (let j = 0; j < n; j++) {
                        B[i][j] += rho * (s[i] * term1[j] + term2[i] * s[j]) -
                                  rho * rho * s[i] * s[j] * sBs;
                    }
                }
                
                x = xNew;
                gradient = gradientNew;
                history.push([...x]);
            }
            
            return {
                solution: x,
                value: f(x),
                iterations: history.length,
                history: history
            };
        }
        
        // Вспомогательные методы
        solveLinearSystem(A, b) {
            // Простая реализация метода Гаусса
            const n = A.length;
            const augmented = A.map((row, i) => [...row, b[i]]);
            
            // Прямой ход
            for (let i = 0; i < n; i++) {
                // Поиск максимального элемента в столбце
                let maxRow = i;
                for (let j = i + 1; j < n; j++) {
                    if (Math.abs(augmented[j][i]) > Math.abs(augmented[maxRow][i])) {
                        maxRow = j;
                    }
                }
                
                // Обмен строк
                [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
                
                // Приведение к треугольному виду
                for (let j = i + 1; j < n; j++) {
                    const factor = augmented[j][i] / augmented[i][i];
                    for (let k = i; k <= n; k++) {
                        augmented[j][k] -= factor * augmented[i][k];
                    }
                }
            }
            
            // Обратный ход
            const x = new Array(n).fill(0);
            for (let i = n - 1; i >= 0; i--) {
                x[i] = augmented[i][n];
                for (let j = i + 1; j < n; j++) {
                    x[i] -= augmented[i][j] * x[j];
                }
                x[i] /= augmented[i][i];
            }
            
            return x;
        }
        
        approximateHessianVector(f, x, v, h = 0.001) {
            // Численное вычисление Hv через разности
            const n = x.length;
            const result = new Array(n).fill(0);
            
            for (let i = 0; i < n; i++) {
                const xPlus = [...x];
                const xMinus = [...x];
                
                for (let j = 0; j < n; j++) {
                    xPlus[j] += h * v[j];
                    xMinus[j] -= h * v[j];
                }
                
                const gradPlus = this.numericalGradient(f, xPlus, h);
                const gradMinus = this.numericalGradient(f, xMinus, h);
                
                result[i] = (gradPlus[i] - gradMinus[i]) / (2 * h);
            }
            
            return result;
        }
        
        numericalGradient(f, x, h = 0.001) {
            const n = x.length;
            const grad = new Array(n);
            
            for (let i = 0; i < n; i++) {
                const xPlus = [...x];
                const xMinus = [...x];
                
                xPlus[i] += h;
                xMinus[i] -= h;
                
                grad[i] = (f(xPlus) - f(xMinus)) / (2 * h);
            }
            
            return grad;
        }
        
        identityMatrix(n) {
            return Array(n).fill().map((_, i) => 
                Array(n).fill().map((_, j) => i === j ? 1 : 0)
            );
        }
        
        matrixVectorMultiply(A, v) {
            return A.map(row => 
                row.reduce((sum, aij, j) => sum + aij * v[j], 0)
            );
        }
    }

    // Экспорт классов
    return {
        NumericalDifferentiation,
        NumericalIntegration,
        ODESolver,
        OptimizationMethods,
        
        // Практические примеры
        examples: {
            // Производная сложной функции
            complexDerivative: function() {
                const f = (x) => Math.sin(x * x) * Math.exp(-x);
                const df = (x) => {
                    const term1 = 2 * x * Math.cos(x * x) * Math.exp(-x);
                    const term2 = -Math.sin(x * x) * Math.exp(-x);
                    return term1 + term2;
                };
                
                return { f, df };
            },
            
            // Градиент функции Розенброка
            rosenbrockGradient: function() {
                // Функция Розенброка: f(x,y) = (1-x)² + 100(y-x²)²
                const f = (x, y) => (1 - x)*(1 - x) + 100*(y - x*x)*(y - x*x);
                const grad = (x, y) => [
                    -2*(1 - x) - 400*x*(y - x*x),
                    200*(y - x*x)
                ];
                
                return { f, grad };
            },
            
            // Уравнение Лотки-Вольтерры
            lotkaVolterra: function() {
                // dx/dt = αx - βxy
                // dy/dt = δxy - γy
                const alpha = 1.1, beta = 0.4, delta = 0.1, gamma = 0.4;
                
                const F = (t, [x, y]) => [
                    alpha * x - beta * x * y,
                    delta * x * y - gamma * y
                ];
                
                return F;
            },
            
            // Гармонический осциллятор с затуханием
            dampedHarmonicOscillator: function() {
                // x'' + 2ζωx' + ω²x = 0
                const omega = 1.0, zeta = 0.1;
                
                const F = (t, [x, v]) => [
                    v,
                    -2*zeta*omega*v - omega*omega*x
                ];
                
                return F;
            }
        },
        
        // Тестирование методов
        testMethods: function() {
            const results = {};
            
            // Тестирование дифференцирования
            const diff = new NumericalDifferentiation();
            const f = Math.sin;
            const x = Math.PI/4;
            const exact = Math.cos(x);
            
            results.differentiation = {};
            for (const [method, func] of Object.entries(diff.methods)) {
                if (typeof func === 'function') {
                    const approx = func(f, x);
                    results.differentiation[method] = {
                        approximation: approx,
                        error: Math.abs(approx - exact),
                        relativeError: Math.abs(approx - exact) / Math.abs(exact)
                    };
                }
            }
            
            // Тестирование интегрирования
            const integ = new NumericalIntegration();
            const g = Math.sin;
            const a = 0, b = Math.PI;
            const exactIntegral = 2;
            
            results.integration = {};
            for (const [method, func] of Object.entries(integ.methods)) {
                if (typeof func === 'function') {
                    const approx = func(g, a, b, 100);
                    results.integration[method] = {
                        approximation: approx,
                        error: Math.abs(approx - exactIntegral),
                        relativeError: Math.abs(approx - exactIntegral) / Math.abs(exactIntegral)
                    };
                }
            }
            
            return results;
        },
        
        // Визуализация ошибок
        visualizeErrors: function(f, df, x, hValues = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125]) {
            const diff = new NumericalDifferentiation();
            const exact = df(x);
            
            const data = {
                h: hValues,
                forward: [],
                backward: [],
                central: [],
                fivePoint: []
            };
            
            for (const h of hValues) {
                const forward = diff.forwardDifference(f, x, h);
                const backward = diff.backwardDifference(f, x, h);
                const central = diff.centralDifference(f, x, h);
                const fivePoint = diff.fivePointDifference(f, x, h);
                
                data.forward.push(Math.abs(forward - exact));
                data.backward.push(Math.abs(backward - exact));
                data.central.push(Math.abs(central - exact));
                data.fivePoint.push(Math.abs(fivePoint - exact));
            }
            
            return data;
        }
    };
}

// Создаем глобальный экземпляр
if (typeof window !== 'undefined') {
    window.NumericalMethods = new NumericalMethods();
}

export default NumericalMethods;