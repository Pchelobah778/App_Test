/**
 * Вспомогательные функции и утилиты
 * Математические функции, работа с DOM, анимации, утилиты
 */

class MathUtils {
    constructor() {
        this.constants = {
            PI: Math.PI,
            E: Math.E,
            PHI: (1 + Math.sqrt(5)) / 2,
            SQRT2: Math.SQRT2,
            SQRT1_2: Math.SQRT1_2
        };
        
        this.tolerance = 1e-12;
    }

    /**
     * Основные математические функции
     */
    
    // Округление с заданной точностью
    round(value, decimals = 2) {
        const factor = Math.pow(10, decimals);
        return Math.round(value * factor) / factor;
    }
    
    // Факториал
    factorial(n) {
        if (n < 0) return NaN;
        if (n === 0 || n === 1) return 1;
        
        let result = 1;
        for (let i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }
    
    // Биномиальные коэффициенты
    binomial(n, k) {
        if (k < 0 || k > n) return 0;
        if (k === 0 || k === n) return 1;
        
        let result = 1;
        for (let i = 1; i <= k; i++) {
            result *= (n - k + i) / i;
        }
        return Math.round(result);
    }
    
    // Гамма-функция (приближение Ланцоша)
    gamma(z) {
        const g = 7;
        const p = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7
        ];
        
        if (z < 0.5) {
            return Math.PI / (Math.sin(Math.PI * z) * this.gamma(1 - z));
        }
        
        z -= 1;
        let x = p[0];
        for (let i = 1; i < g + 2; i++) {
            x += p[i] / (z + i);
        }
        
        const t = z + g + 0.5;
        return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
    }
    
    // Бета-функция
    beta(x, y) {
        return this.gamma(x) * this.gamma(y) / this.gamma(x + y);
    }
    
    // Функция ошибок (erf)
    erf(x) {
        // Аппроксимация Абрамовица и Стегун
        const t = 1.0 / (1.0 + 0.5 * Math.abs(x));
        const tau = t * Math.exp(-x*x - 1.26551223 +
            t * (1.00002368 +
            t * (0.37409196 +
            t * (0.09678418 +
            t * (-0.18628806 +
            t * (0.27886807 +
            t * (-1.13520398 +
            t * (1.48851587 +
            t * (-0.82215223 +
            t * 0.17087277)))))))));
        
        return x >= 0 ? 1 - tau : tau - 1;
    }
    
    // Интеграл вероятности
    phi(x) {
        return 0.5 * (1 + this.erf(x / Math.sqrt(2)));
    }
    
    // Сигмоида
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    // Гиперболические функции
    sech(x) { return 1 / Math.cosh(x); }
    csch(x) { return 1 / Math.sinh(x); }
    coth(x) { return 1 / Math.tanh(x); }
    
    // Обратные гиперболические функции
    asinh(x) { return Math.log(x + Math.sqrt(x*x + 1)); }
    acosh(x) { return Math.log(x + Math.sqrt(x*x - 1)); }
    atanh(x) { return 0.5 * Math.log((1 + x) / (1 - x)); }
    
    /**
     * Линейная алгебра
     */
    
    // Скалярное произведение
    dot(a, b) {
        if (a.length !== b.length) throw new Error('Векторы должны иметь одинаковую длину');
        return a.reduce((sum, ai, i) => sum + ai * b[i], 0);
    }
    
    // Векторное произведение (3D)
    cross(a, b) {
        if (a.length !== 3 || b.length !== 3) {
            throw new Error('Векторы должны быть 3-мерными');
        }
        return [
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ];
    }
    
    // Норма вектора
    norm(vector, p = 2) {
        if (p === Infinity) {
            return Math.max(...vector.map(Math.abs));
        }
        return Math.pow(
            vector.reduce((sum, vi) => sum + Math.pow(Math.abs(vi), p), 0),
            1/p
        );
    }
    
    // Нормализация вектора
    normalize(vector) {
        const n = this.norm(vector);
        if (n === 0) return vector;
        return vector.map(v => v / n);
    }
    
    // Угол между векторами
    angleBetween(a, b) {
        const dotProduct = this.dot(a, b);
        const norms = this.norm(a) * this.norm(b);
        return Math.acos(Math.min(1, Math.max(-1, dotProduct / norms)));
    }
    
    // Проекция вектора
    project(a, b) {
        const scalar = this.dot(a, b) / this.dot(b, b);
        return b.map(bi => scalar * bi);
    }
    
    // Отражение вектора
    reflect(v, n) {
        const dot = this.dot(v, n);
        return v.map((vi, i) => vi - 2 * dot * n[i]);
    }
    
    // Матричные операции
    matrixMultiply(A, B) {
        const m = A.length;
        const n = A[0].length;
        const p = B[0].length;
        
        if (n !== B.length) throw new Error('Несовместимые размеры матриц');
        
        const C = Array(m).fill().map(() => Array(p).fill(0));
        
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < p; j++) {
                for (let k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        return C;
    }
    
    // Транспонирование матрицы
    transpose(matrix) {
        const rows = matrix.length;
        const cols = matrix[0].length;
        const result = Array(cols).fill().map(() => Array(rows).fill(0));
        
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        
        return result;
    }
    
    // Определитель матрицы
    determinant(matrix) {
        const n = matrix.length;
        
        if (n === 1) return matrix[0][0];
        if (n === 2) {
            return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0];
        }
        
        let det = 0;
        for (let j = 0; j < n; j++) {
            const minor = this.getMinor(matrix, 0, j);
            det += matrix[0][j] * Math.pow(-1, j) * this.determinant(minor);
        }
        
        return det;
    }
    
    // Минор матрицы
    getMinor(matrix, row, col) {
        return matrix
            .filter((_, i) => i !== row)
            .map(row => row.filter((_, j) => j !== col));
    }
    
    // Обратная матрица
    inverse(matrix) {
        const n = matrix.length;
        const det = this.determinant(matrix);
        
        if (Math.abs(det) < this.tolerance) {
            throw new Error('Матрица вырождена');
        }
        
        const adjugate = Array(n).fill().map(() => Array(n).fill(0));
        
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                const minor = this.getMinor(matrix, i, j);
                adjugate[j][i] = Math.pow(-1, i + j) * this.determinant(minor);
            }
        }
        
        return adjugate.map(row => row.map(val => val / det));
    }
    
    // Собственные значения и векторы (степенной метод)
    powerMethod(matrix, maxIter = 1000, tol = 1e-6) {
        let n = matrix.length;
        let x = Array(n).fill(1);
        x = this.normalize(x);
        
        let lambdaOld = 0;
        
        for (let iter = 0; iter < maxIter; iter++) {
            // Ax
            const Ax = Array(n).fill(0);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    Ax[i] += matrix[i][j] * x[j];
                }
            }
            
            // Новое приближение собственного значения
            const lambda = this.dot(Ax, x) / this.dot(x, x);
            
            // Проверка сходимости
            if (Math.abs(lambda - lambdaOld) < tol) {
                return {
                    eigenvalue: lambda,
                    eigenvector: x,
                    iterations: iter
                };
            }
            
            // Обновление
            x = this.normalize(Ax);
            lambdaOld = lambda;
        }
        
        return {
            eigenvalue: lambdaOld,
            eigenvector: x,
            iterations: maxIter,
            converged: false
        };
    }
    
    /**
     * Интерполяция и аппроксимация
     */
    
    // Линейная интерполяция
    lerp(a, b, t) {
        return a + (b - a) * t;
    }
    
    // Кубическая интерполяция Эрмита
    smoothstep(t) {
        return t * t * (3 - 2 * t);
    }
    
    // Smootherstep (Кеньон Перлин)
    smootherstep(t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }
    
    // Интерполяция Лагранжа
    lagrangeInterpolation(points, x) {
        let result = 0;
        const n = points.length;
        
        for (let i = 0; i < n; i++) {
            const [xi, yi] = points[i];
            let term = yi;
            
            for (let j = 0; j < n; j++) {
                if (i !== j) {
                    const [xj] = points[j];
                    term *= (x - xj) / (xi - xj);
                }
            }
            
            result += term;
        }
        
        return result;
    }
    
    // Полином Ньютона
    newtonInterpolation(points) {
        const n = points.length;
        const dividedDiffs = [points.map(p => p[1])];
        
        // Разделенные разности
        for (let i = 1; i < n; i++) {
            dividedDiffs[i] = [];
            for (let j = 0; j < n - i; j++) {
                dividedDiffs[i][j] = (dividedDiffs[i-1][j+1] - dividedDiffs[i-1][j]) / 
                                   (points[j+i][0] - points[j][0]);
            }
        }
        
        // Функция интерполяции
        return (x) => {
            let result = dividedDiffs[0][0];
            let product = 1;
            
            for (let i = 1; i < n; i++) {
                product *= (x - points[i-1][0]);
                result += dividedDiffs[i][0] * product;
            }
            
            return result;
        };
    }
    
    // Кривая Безье
    bezierCurve(points, t) {
        const n = points.length - 1;
        let result = [0, 0];
        
        for (let i = 0; i <= n; i++) {
            const binom = this.binomial(n, i);
            const weight = binom * Math.pow(1 - t, n - i) * Math.pow(t, i);
            result[0] += weight * points[i][0];
            result[1] += weight * points[i][1];
        }
        
        return result;
    }
    
    // Сплайн-интерполяция (кубические сплайны)
    cubicSpline(points) {
        const n = points.length - 1;
        const h = Array(n).fill(0);
        const alpha = Array(n).fill(0);
        const l = Array(n + 1).fill(1);
        const mu = Array(n).fill(0);
        const z = Array(n + 1).fill(0);
        
        const x = points.map(p => p[0]);
        const a = points.map(p => p[1]);
        const b = Array(n).fill(0);
        const c = Array(n + 1).fill(0);
        const d = Array(n).fill(0);
        
        // Шаги
        for (let i = 0; i < n; i++) {
            h[i] = x[i + 1] - x[i];
        }
        
        // Альфа
        for (let i = 1; i < n; i++) {
            alpha[i] = (3/h[i])*(a[i+1] - a[i]) - (3/h[i-1])*(a[i] - a[i-1]);
        }
        
        // Прогонка
        for (let i = 1; i < n; i++) {
            l[i] = 2*(x[i+1] - x[i-1]) - h[i-1]*mu[i-1];
            mu[i] = h[i]/l[i];
            z[i] = (alpha[i] - h[i-1]*z[i-1])/l[i];
        }
        
        // Коэффициенты
        for (let j = n - 1; j >= 0; j--) {
            c[j] = z[j] - mu[j]*c[j+1];
            b[j] = (a[j+1] - a[j])/h[j] - h[j]*(c[j+1] + 2*c[j])/3;
            d[j] = (c[j+1] - c[j])/(3*h[j]);
        }
        
        // Функция интерполяции
        return (xVal) => {
            let i = 0;
            while (i < n && xVal > x[i+1]) i++;
            
            const dx = xVal - x[i];
            return a[i] + b[i]*dx + c[i]*dx*dx + d[i]*dx*dx*dx;
        };
    }
    
    /**
     * Статистика и вероятность
     */
    
    // Среднее значение
    mean(values) {
        return values.reduce((sum, val) => sum + val, 0) / values.length;
    }
    
    // Медиана
    median(values) {
        const sorted = [...values].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        
        if (sorted.length % 2 === 0) {
            return (sorted[mid - 1] + sorted[mid]) / 2;
        } else {
            return sorted[mid];
        }
    }
    
    // Дисперсия
    variance(values, isSample = true) {
        const avg = this.mean(values);
        const sum = values.reduce((sum, val) => sum + Math.pow(val - avg, 2), 0);
        return sum / (values.length - (isSample ? 1 : 0));
    }
    
    // Стандартное отклонение
    stdDev(values, isSample = true) {
        return Math.sqrt(this.variance(values, isSample));
    }
    
    // Ковариация
    covariance(x, y, isSample = true) {
        if (x.length !== y.length) {
            throw new Error('Массивы должны иметь одинаковую длину');
        }
        
        const meanX = this.mean(x);
        const meanY = this.mean(y);
        const sum = x.reduce((sum, xi, i) => sum + (xi - meanX) * (y[i] - meanY), 0);
        
        return sum / (x.length - (isSample ? 1 : 0));
    }
    
    // Корреляция (Пирсон)
    correlation(x, y) {
        const cov = this.covariance(x, y);
        const stdX = this.stdDev(x);
        const stdY = this.stdDev(y);
        
        return cov / (stdX * stdY);
    }
    
    // Распределения
    
    // Нормальное распределение
    normalPDF(x, mu = 0, sigma = 1) {
        return (1 / (sigma * Math.sqrt(2 * Math.PI))) * 
               Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2));
    }
    
    // Распределение Стьюдента
    tPDF(x, df) {
        const coef = this.gamma((df + 1) / 2) / 
                    (Math.sqrt(df * Math.PI) * this.gamma(df / 2));
        return coef * Math.pow(1 + x*x/df, -(df + 1)/2);
    }
    
    // Распределение χ²
    chi2PDF(x, k) {
        if (x <= 0) return 0;
        return Math.pow(x, k/2 - 1) * Math.exp(-x/2) / 
               (Math.pow(2, k/2) * this.gamma(k/2));
    }
    
    // Распределение Фишера
    fPDF(x, d1, d2) {
        if (x <= 0) return 0;
        const numerator = Math.pow(d1*x, d1) * Math.pow(d2, d2);
        const denominator = Math.pow(d1*x + d2, d1 + d2);
        const beta = this.gamma(d1/2) * this.gamma(d2/2) / this.gamma((d1 + d2)/2);
        
        return Math.sqrt(numerator/denominator) / (x * beta);
    }
    
    /**
     * Оптимизация и поиск корней
     */
    
    // Метод бисекции
    bisection(f, a, b, tol = 1e-12, maxIter = 1000) {
        if (f(a) * f(b) > 0) {
            throw new Error('Функция должна иметь разные знаки на концах интервала');
        }
        
        let iter = 0;
        while ((b - a) / 2 > tol && iter < maxIter) {
            const c = (a + b) / 2;
            if (f(c) === 0) return c;
            
            if (f(a) * f(c) < 0) {
                b = c;
            } else {
                a = c;
            }
            iter++;
        }
        
        return (a + b) / 2;
    }
    
    // Метод Ньютона для поиска корней
    newtonRoot(f, df, x0, tol = 1e-12, maxIter = 1000) {
        let x = x0;
        let iter = 0;
        
        while (iter < maxIter) {
            const fx = f(x);
            const dfx = df(x);
            
            if (Math.abs(fx) < tol) return x;
            if (Math.abs(dfx) < this.tolerance) {
                throw new Error('Производная слишком мала');
            }
            
            const xNext = x - fx / dfx;
            
            if (Math.abs(xNext - x) < tol) return xNext;
            
            x = xNext;
            iter++;
        }
        
        return x;
    }
    
    // Метод секущих
    secantMethod(f, x0, x1, tol = 1e-12, maxIter = 1000) {
        let xPrev = x0;
        let x = x1;
        let iter = 0;
        
        while (iter < maxIter) {
            const fPrev = f(xPrev);
            const fCurr = f(x);
            
            if (Math.abs(fCurr) < tol) return x;
            
            const xNext = x - fCurr * (x - xPrev) / (fCurr - fPrev);
            
            if (Math.abs(xNext - x) < tol) return xNext;
            
            xPrev = x;
            x = xNext;
            iter++;
        }
        
        return x;
    }
    
    // Метод золотого сечения для оптимизации
    goldenSection(f, a, b, tol = 1e-12, maxIter = 1000, maximize = false) {
        const phi = (1 + Math.sqrt(5)) / 2;
        const resphi = 2 - phi;
        
        let x1 = a + resphi * (b - a);
        let x2 = b - resphi * (b - a);
        let f1 = f(x1);
        let f2 = f(x2);
        
        for (let iter = 0; iter < maxIter; iter++) {
            if (Math.abs(b - a) < tol) break;
            
            if ((!maximize && f1 < f2) || (maximize && f1 > f2)) {
                b = x2;
                x2 = x1;
                f2 = f1;
                x1 = a + resphi * (b - a);
                f1 = f(x1);
            } else {
                a = x1;
                x1 = x2;
                f1 = f2;
                x2 = b - resphi * (b - a);
                f2 = f(x2);
            }
        }
        
        return (a + b) / 2;
    }
    
    /**
     * Преобразования координат
     */
    
    // Декартовы в полярные
    cartesianToPolar(x, y) {
        return {
            r: Math.sqrt(x*x + y*y),
            theta: Math.atan2(y, x)
        };
    }
    
    // Полярные в декартовы
    polarToCartesian(r, theta) {
        return {
            x: r * Math.cos(theta),
            y: r * Math.sin(theta)
        };
    }
    
    // Декартовы в сферические
    cartesianToSpherical(x, y, z) {
        const r = Math.sqrt(x*x + y*y + z*z);
        return {
            r: r,
            theta: Math.acos(z / r),
            phi: Math.atan2(y, x)
        };
    }
    
    // Сферические в декартовы
    sphericalToCartesian(r, theta, phi) {
        return {
            x: r * Math.sin(theta) * Math.cos(phi),
            y: r * Math.sin(theta) * Math.sin(phi),
            z: r * Math.cos(theta)
        };
    }
    
    /**
     * Геометрия
     */
    
    // Расстояние между точками
    distance(p1, p2) {
        return Math.sqrt(
            p1.reduce((sum, val, i) => sum + Math.pow(val - p2[i], 2), 0)
        );
    }
    
    // Площадь треугольника по координатам
    triangleArea(p1, p2, p3) {
        const a = this.distance(p1, p2);
        const b = this.distance(p2, p3);
        const c = this.distance(p3, p1);
        const s = (a + b + c) / 2;
        return Math.sqrt(s * (s - a) * (s - b) * (s - c));
    }
    
    // Объем тетраэдра
    tetrahedronVolume(p1, p2, p3, p4) {
        const mat = [
            [p1[0], p1[1], p1[2], 1],
            [p2[0], p2[1], p2[2], 1],
            [p3[0], p3[1], p3[2], 1],
            [p4[0], p4[1], p4[2], 1]
        ];
        
        return Math.abs(this.determinant(mat)) / 6;
    }
    
    // Проверка на выпуклость полигона
    isConvex(polygon) {
        const n = polygon.length;
        if (n < 3) return false;
        
        let sign = 0;
        for (let i = 0; i < n; i++) {
            const p1 = polygon[i];
            const p2 = polygon[(i + 1) % n];
            const p3 = polygon[(i + 2) % n];
            
            const cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - 
                         (p2[1] - p1[1]) * (p3[0] - p2[0]);
            
            if (cross !== 0) {
                if (sign === 0) {
                    sign = cross > 0 ? 1 : -1;
                } else if (sign * cross < 0) {
                    return false;
                }
            }
        }
        
        return true;
    }
}

class DOMUtils {
    constructor() {
        this.observers = new Map();
    }
    
    /**
     * Создание элементов
     */
    
    // Создание элемента с атрибутами
    createElement(tag, attributes = {}, children = []) {
        const element = document.createElement(tag);
        
        // Атрибуты
        for (const [key, value] of Object.entries(attributes)) {
            if (key === 'className') {
                element.className = value;
            } else if (key === 'innerHTML') {
                element.innerHTML = value;
            } else if (key === 'textContent') {
                element.textContent = value;
            } else if (key.startsWith('data-')) {
                element.setAttribute(key, value);
            } else if (key.startsWith('on')) {
                element.addEventListener(key.substring(2).toLowerCase(), value);
            } else {
                element.setAttribute(key, value);
            }
        }
        
        // Дочерние элементы
        if (Array.isArray(children)) {
            children.forEach(child => {
                if (typeof child === 'string') {
                    element.appendChild(document.createTextNode(child));
                } else if (child instanceof Node) {
                    element.appendChild(child);
                }
            });
        }
        
        return element;
    }
    
    // Создание SVG элемента
    createSVG(tag, attributes = {}) {
        const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
        
        for (const [key, value] of Object.entries(attributes)) {
            element.setAttribute(key, value);
        }
        
        return element;
    }
    
    // Создание математической формулы
    createMathFormula(latex, options = {}) {
        const container = this.createElement('div', {
            className: 'math-container'
        });
        
        // Используем MathJax если доступен
        if (window.MathJax) {
            container.innerHTML = `\\[${latex}\\]`;
            MathJax.typeset([container]);
        } else {
            container.textContent = latex;
        }
        
        return container;
    }
    
    /**
     * Анимации
     */
    
    // Плавное появление
    fadeIn(element, duration = 300) {
        element.style.opacity = '0';
        element.style.display = 'block';
        
        let start = null;
        const animate = (timestamp) => {
            if (!start) start = timestamp;
            const progress = timestamp - start;
            const opacity = Math.min(progress / duration, 1);