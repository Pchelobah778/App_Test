/**
 * Векторный анализ: градиент, дивергенция, ротор
 * 3D визуализации и вычисления
 */

class VectorAnalysis {
    constructor() {
        this.canvas3D = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        
        // Цветовые схемы
        this.colorSchemes = {
            gradient: ['#4f46e5', '#7c3aed', '#a855f7'],
            divergence: ['#dc2626', '#ea580c', '#f59e0b'],
            curl: ['#059669', '#0d9488', '#14b8a6']
        };
    }

    /**
     * Градиент скалярного поля
     */
    class Gradient {
        constructor() {
            this.functions = {
                // 2D функции
                paraboloid: (x, y) => x*x + y*y,
                saddle: (x, y) => x*x - y*y,
                gaussian: (x, y) => Math.exp(-(x*x + y*y)),
                ripple: (x, y) => Math.sin(x)*Math.cos(y),
                mexicanHat: (x, y) => (1 - (x*x + y*y)) * Math.exp(-(x*x + y*y)/2),
                
                // 3D функции
                sphere3D: (x, y, z) => x*x + y*y + z*z,
                hyperboloid: (x, y, z) => x*x + y*y - z*z,
                wave3D: (x, y, z) => Math.sin(x)*Math.cos(y)*Math.exp(-z*z)
            };
        }
        
        // Численный расчет градиента
        calculate(f, point, h = 0.0001) {
            const dim = point.length;
            const gradient = new Array(dim);
            
            for (let i = 0; i < dim; i++) {
                const pointPlus = [...point];
                const pointMinus = [...point];
                
                pointPlus[i] += h;
                pointMinus[i] -= h;
                
                gradient[i] = (f(...pointPlus) - f(...pointMinus)) / (2 * h);
            }
            
            return gradient;
        }
        
        // Аналитический градиент для известных функций
        analyticGradient(funcName, point) {
            switch(funcName) {
                case 'paraboloid':
                    return [2*point[0], 2*point[1]];
                case 'saddle':
                    return [2*point[0], -2*point[1]];
                case 'gaussian':
                    const exp = Math.exp(-(point[0]*point[0] + point[1]*point[1]));
                    return [-2*point[0]*exp, -2*point[1]*exp];
                case 'ripple':
                    return [Math.cos(point[0])*Math.cos(point[1]), 
                           -Math.sin(point[0])*Math.sin(point[1])];
                case 'mexicanHat':
                    const r2 = point[0]*point[0] + point[1]*point[1];
                    const exp2 = Math.exp(-r2/2);
                    return [point[0]*(r2 - 3)*exp2, point[1]*(r2 - 3)*exp2];
                default:
                    return this.calculate(this.functions[funcName], point);
            }
        }
        
        // Уровневые линии (изолинии)
        getContourLines(funcName, xRange, yRange, levels = 10) {
            const f = this.functions[funcName];
            const lines = [];
            const [xMin, xMax] = xRange;
            const [yMin, yMax] = yRange;
            
            // Находим мин и макс значения функции
            let fMin = Infinity;
            let fMax = -Infinity;
            
            for (let x = xMin; x <= xMax; x += (xMax - xMin)/20) {
                for (let y = yMin; y <= yMax; y += (yMax - yMin)/20) {
                    const val = f(x, y);
                    fMin = Math.min(fMin, val);
                    fMax = Math.max(fMax, val);
                }
            }
            
            // Создаем уровни
            for (let i = 0; i < levels; i++) {
                const level = fMin + (fMax - fMin) * i / (levels - 1);
                const line = this.traceContour(f, xRange, yRange, level, 100);
                lines.push({level, points: line});
            }
            
            return lines;
        }
        
        // Трассировка изолинии (марширующие квадраты)
        traceContour(f, xRange, yRange, level, resolution) {
            const [xMin, xMax] = xRange;
            const [yMin, yMax] = yRange;
            const dx = (xMax - xMin) / resolution;
            const dy = (yMax - yMin) / resolution;
            
            const points = [];
            
            for (let i = 0; i < resolution; i++) {
                for (let j = 0; j < resolution; j++) {
                    const x = xMin + i * dx;
                    const y = yMin + j * dy;
                    
                    // Значения в углах квадрата
                    const f00 = f(x, y) - level;
                    const f10 = f(x + dx, y) - level;
                    const f01 = f(x, y + dy) - level;
                    const f11 = f(x + dx, y + dy) - level;
                    
                    // Определяем, пересекает ли контур квадрат
                    const config = (f00 > 0 ? 8 : 0) |
                                  (f10 > 0 ? 4 : 0) |
                                  (f01 > 0 ? 2 : 0) |
                                  (f11 > 0 ? 1 : 0);
                    
                    if (config === 0 || config === 15) continue;
                    
                    // Линейная интерполяция
                    let x1, y1, x2, y2;
                    
                    switch(config) {
                        case 1: case 14:
                            x1 = x + dx * f00/(f00 - f10);
                            y1 = y;
                            x2 = x + dx;
                            y2 = y + dy * f10/(f10 - f11);
                            break;
                        case 2: case 13:
                            x1 = x;
                            y1 = y + dy * f00/(f00 - f01);
                            x2 = x + dx * f01/(f01 - f11);
                            y2 = y + dy;
                            break;
                        // ... другие случаи
                    }
                    
                    if (x1 && y1 && x2 && y2) {
                        points.push([[x1, y1], [x2, y2]]);
                    }
                }
            }
            
            return points;
        }
        
        // Визуализация градиентного поля
        createGradientField(funcName, xRange, yRange, resolution = 20) {
            const field = [];
            const [xMin, xMax] = xRange;
            const [yMin, yMax] = yRange;
            const dx = (xMax - xMin) / resolution;
            const dy = (yMax - yMin) / resolution;
            
            for (let i = 0; i <= resolution; i++) {
                for (let j = 0; j <= resolution; j++) {
                    const x = xMin + i * dx;
                    const y = yMin + j * dy;
                    const grad = this.analyticGradient(funcName, [x, y]);
                    
                    // Нормализуем для визуализации
                    const magnitude = Math.sqrt(grad[0]*grad[0] + grad[1]*grad[1]);
                    const scale = 0.3 * Math.min(dx, dy) / (magnitude + 0.001);
                    
                    field.push({
                        position: [x, y],
                        vector: [grad[0] * scale, grad[1] * scale],
                        magnitude: magnitude,
                        color: this.getColorByMagnitude(magnitude)
                    });
                }
            }
            
            return field;
        }
        
        getColorByMagnitude(magnitude) {
            const maxMag = 5; // максимальная величина для цветовой шкалы
            const t = Math.min(magnitude / maxMag, 1);
            return d3.interpolateRgb('#4f46e5', '#f59e0b')(t);
        }
    }

    /**
     * Дивергенция векторного поля
     */
    class Divergence {
        constructor() {
            this.fields = {
                // 2D поля
                radial: (x, y) => [x, y],                     // div = 2
                vortex: (x, y) => [-y, x],                    // div = 0
                sink: (x, y) => [-x, -y],                     // div = -2
                shear: (x, y) => [y, 0],                      // div = 0
                compressible: (x, y) => [x*x, y*y],           // div = 2x + 2y
                
                // 3D поля
                radial3D: (x, y, z) => [x, y, z],             // div = 3
                vortex3D: (x, y, z) => [-y, x, 0],            // div = 0
                magneticDipole: (x, y, z) => {
                    const r = Math.sqrt(x*x + y*y + z*z) + 0.001;
                    return [(3*x*z)/(r*r*r*r*r), (3*y*z)/(r*r*r*r*r), 
                           (3*z*z - r*r)/(r*r*r*r*r)];
                }                                             // div = 0 (соленоидальное)
            };
        }
        
        // Численный расчет дивергенции
        calculate(field, point, h = 0.0001) {
            const dim = point.length;
            let div = 0;
            
            for (let i = 0; i < dim; i++) {
                const pointPlus = [...point];
                const pointMinus = [...point];
                
                pointPlus[i] += h;
                pointMinus[i] -= h;
                
                const fPlus = field(...pointPlus);
                const fMinus = field(...pointMinus);
                
                div += (fPlus[i] - fMinus[i]) / (2 * h);
            }
            
            return div;
        }
        
        // Аналитическая дивергенция для известных полей
        analyticDivergence(fieldName, point) {
            const [x, y, z = 0] = point;
            
            switch(fieldName) {
                case 'radial':
                    return 2;
                case 'vortex':
                    return 0;
                case 'sink':
                    return -2;
                case 'shear':
                    return 0;
                case 'compressible':
                    return 2*x + 2*y;
                case 'radial3D':
                    return 3;
                case 'vortex3D':
                    return 0;
                case 'magneticDipole':
                    return 0; // всегда соленоидальное
                default:
                    return this.calculate(this.fields[fieldName], point);
            }
        }
        
        // Поток через поверхность (численная интеграция)
        fluxThroughSurface(fieldName, surface, resolution = 50) {
            let totalFlux = 0;
            
            // Для каждой грани поверхности
            surface.faces.forEach(face => {
                // Нормаль к поверхности
                const normal = this.calculateFaceNormal(face);
                
                // Интеграция методом средних прямоугольников
                const area = this.calculateFaceArea(face);
                const center = this.calculateFaceCenter(face);
                
                const fieldValue = this.fields[fieldName](...center);
                const dotProduct = this.dot(fieldValue, normal);
                
                totalFlux += dotProduct * area;
            });
            
            return totalFlux;
        }
        
        // Теорема Гаусса (проверка)
        gaussTheorem(fieldName, volume, surface, resolution = 50) {
            // Поток через поверхность
            const flux = this.fluxThroughSurface(fieldName, surface, resolution);
            
            // Интеграл дивергенции по объему
            let volumeIntegral = 0;
            const voxels = this.discretizeVolume(volume, resolution);
            
            voxels.forEach(voxel => {
                const div = this.analyticDivergence(fieldName, voxel.center);
                volumeIntegral += div * voxel.volume;
            });
            
            return {
                flux,
                volumeIntegral,
                difference: Math.abs(flux - volumeIntegral),
                relativeError: Math.abs(flux - volumeIntegral) / Math.max(Math.abs(flux), 0.001)
            };
        }
        
        // Визуализация дивергенции
        createDivergenceField(fieldName, xRange, yRange, resolution = 20) {
            const field = [];
            const [xMin, xMax] = xRange;
            const [yMin, yMax] = yRange;
            const dx = (xMax - xMin) / resolution;
            const dy = (yMax - yMin) / resolution;
            
            for (let i = 0; i <= resolution; i++) {
                for (let j = 0; j <= resolution; j++) {
                    const x = xMin + i * dx;
                    const y = yMin + j * dy;
                    
                    // Векторное поле
                    const vec = this.fields[fieldName](x, y);
                    
                    // Дивергенция
                    const div = this.analyticDivergence(fieldName, [x, y]);
                    
                    // Масштабирование векторов
                    const magnitude = Math.sqrt(vec[0]*vec[0] + vec[1]*vec[1]);
                    const scale = 0.3 * Math.min(dx, dy) / (magnitude + 0.001);
                    
                    field.push({
                        position: [x, y],
                        vector: [vec[0] * scale, vec[1] * scale],
                        divergence: div,
                        color: this.getColorByDivergence(div)
                    });
                }
            }
            
            return field;
        }
        
        getColorByDivergence(div) {
            // Красный для положительной (источник), синий для отрицательной (сток)
            const maxDiv = 3;
            const t = Math.max(-1, Math.min(1, div / maxDiv));
            
            if (t > 0) {
                return d3.interpolateRgb('#ffffff', '#dc2626')(t);
            } else {
                return d3.interpolateRgb('#ffffff', '#1d4ed8')(-t);
            }
        }
        
        // Вспомогательные геометрические функции
        dot(a, b) {
            return a.reduce((sum, ai, i) => sum + ai * b[i], 0);
        }
        
        calculateFaceNormal(face) {
            // Для треугольника
            if (face.length === 3) {
                const [p0, p1, p2] = face;
                const v1 = p1.map((v, i) => v - p0[i]);
                const v2 = p2.map((v, i) => v - p0[i]);
                
                // Векторное произведение
                const normal = [
                    v1[1]*v2[2] - v1[2]*v2[1],
                    v1[2]*v2[0] - v1[0]*v2[2],
                    v1[0]*v2[1] - v1[1]*v2[0]
                ];
                
                // Нормализация
                const length = Math.sqrt(normal.reduce((sum, n) => sum + n*n, 0));
                return normal.map(n => n / length);
            }
            return [0, 0, 1]; // По умолчанию
        }
        
        calculateFaceArea(face) {
            if (face.length === 3) {
                const [p0, p1, p2] = face;
                const a = Math.sqrt(this.distanceSquared(p0, p1));
                const b = Math.sqrt(this.distanceSquared(p1, p2));
                const c = Math.sqrt(this.distanceSquared(p2, p0));
                const s = (a + b + c) / 2;
                return Math.sqrt(s * (s - a) * (s - b) * (s - c));
            }
            return 1; // По умолчанию
        }
        
        distanceSquared(p1, p2) {
            return p1.reduce((sum, v, i) => sum + (v - p2[i])*(v - p2[i]), 0);
        }
        
        calculateFaceCenter(face) {
            const sum = face.reduce((acc, point) => {
                return acc.map((v, i) => v + point[i]);
            }, new Array(face[0].length).fill(0));
            
            return sum.map(v => v / face.length);
        }
        
        discretizeVolume(volume, resolution) {
            const voxels = [];
            const [xMin, xMax, yMin, yMax, zMin, zMax] = volume;
            
            const dx = (xMax - xMin) / resolution;
            const dy = (yMax - yMin) / resolution;
            const dz = (zMax - zMin) / resolution;
            
            for (let i = 0; i < resolution; i++) {
                for (let j = 0; j < resolution; j++) {
                    for (let k = 0; k < resolution; k++) {
                        const x = xMin + (i + 0.5) * dx;
                        const y = yMin + (j + 0.5) * dy;
                        const z = zMin + (k + 0.5) * dz;
                        
                        voxels.push({
                            center: [x, y, z],
                            volume: dx * dy * dz
                        });
                    }
                }
            }
            
            return voxels;
        }
    }

    /**
     * Ротор (вихрь) векторного поля
     */
    class Curl {
        constructor() {
            this.fields = {
                // 2D поля (ротор в направлении z)
                vortex2D: (x, y) => [-y, x],                    // rot = 2k
                shear2D: (x, y) => [y, 0],                      // rot = -k
                compressibleVortex: (x, y) => [-y/(x*x+y*y+0.01), x/(x*x+y*y+0.01)],
                
                // 3D поля
                vortex3D: (x, y, z) => [-y, x, 0],             // rot = [0, 0, 2]
                helical: (x, y, z) => [-y, x, 1],              // спиральное поле
                twisted: (x, y, z) => [-z, 0, x],              // rot = [0, -2, 0]
                solenoidal: (x, y, z) => {
                    // Поле с нулевой дивергенцией и ненулевым ротором
                    return [-y*z, x*z, 0];
                }
            };
        }
        
        // Численный расчет ротора (3D)
        calculate(field, point, h = 0.0001) {
            const [x, y, z] = point;
            
            // dFz/dy - dFy/dz
            const rotX = (this.partialDerivative(field, point, 2, 1, h) -
                         this.partialDerivative(field, point, 1, 2, h));
            
            // dFx/dz - dFz/dx
            const rotY = (this.partialDerivative(field, point, 0, 2, h) -
                         this.partialDerivative(field, point, 2, 0, h));
            
            // dFy/dx - dFx/dy
            const rotZ = (this.partialDerivative(field, point, 1, 0, h) -
                         this.partialDerivative(field, point, 0, 1, h));
            
            return [rotX, rotY, rotZ];
        }
        
        // Частная производная ∂F_i/∂x_j
        partialDerivative(field, point, i, j, h) {
            const pointPlus = [...point];
            const pointMinus = [...point];
            
            pointPlus[j] += h;
            pointMinus[j] -= h;
            
            const fPlus = field(...pointPlus);
            const fMinus = field(...pointMinus);
            
            return (fPlus[i] - fMinus[i]) / (2 * h);
        }
        
        // Аналитический ротор для известных полей
        analyticCurl(fieldName, point) {
            const [x, y, z = 0] = point;
            
            switch(fieldName) {
                case 'vortex2D':
                    return [0, 0, 2]; // Вращение вокруг оси z
                case 'shear2D':
                    return [0, 0, -1];
                case 'vortex3D':
                    return [0, 0, 2];
                case 'helical':
                    return [0, 0, 2]; // Тоже 2k, но с добавленной z-компонентой
                case 'twisted':
                    return [0, -2, 0];
                case 'solenoidal':
                    return [x, y, 2*z];
                default:
                    return this.calculate(this.fields[fieldName], point);
            }
        }
        
        // Циркуляция по контуру (численная интеграция)
        circulation(fieldName, contour, resolution = 100) {
            let circ = 0;
            const n = contour.length;
            
            for (let i = 0; i < n; i++) {
                const p1 = contour[i];
                const p2 = contour[(i + 1) % n];
                
                // Средняя точка отрезка
                const mid = p1.map((v, idx) => (v + p2[idx]) / 2);
                
                // Вектор поля в средней точке
                const fieldValue = this.fields[fieldName](...mid);
                
                // Касательный вектор
                const tangent = p2.map((v, idx) => v - p1[idx]);
                
                // Скалярное произведение
                const dot = fieldValue.reduce((sum, f, idx) => sum + f * tangent[idx], 0);
                
                circ += dot;
            }
            
            return circ;
        }
        
        // Теорема Стокса (проверка)
        stokesTheorem(fieldName, contour, surface, resolution = 50) {
            // Циркуляция по контуру
            const circulation = this.circulation(fieldName, contour, resolution);
            
            // Поток ротора через поверхность
            let flux = 0;
            
            // Дискретизация поверхности
            const triangles = this.triangulateSurface(surface, resolution);
            
            triangles.forEach(triangle => {
                const center = triangle.reduce((acc, p) => {
                    return acc.map((v, i) => v + p[i]);
                }, [0, 0, 0]).map(v => v / 3);
                
                const normal = this.calculateTriangleNormal(triangle);
                const area = this.calculateTriangleArea(triangle);
                
                const curl = this.analyticCurl(fieldName, center);
                const dot = curl.reduce((sum, c, i) => sum + c * normal[i], 0);
                
                flux += dot * area;
            });
            
            return {
                circulation,
                flux,
                difference: Math.abs(circulation - flux),
                relativeError: Math.abs(circulation - flux) / Math.max(Math.abs(circulation), 0.001)
            };
        }
        
        // Визуализация ротора
        createCurlField(fieldName, xRange, yRange, z = 0, resolution = 20) {
            const field = [];
            const [xMin, xMax] = xRange;
            const [yMin, yMax] = yRange;
            const dx = (xMax - xMin) / resolution;
            const dy = (yMax - yMin) / resolution;
            
            for (let i = 0; i <= resolution; i++) {
                for (let j = 0; j <= resolution; j++) {
                    const x = xMin + i * dx;
                    const y = yMin + j * dy;
                    const point = [x, y, z];
                    
                    // Векторное поле
                    const vec = this.fields[fieldName](...point);
                    
                    // Ротор
                    const curl = this.analyticCurl(fieldName, point);
                    const curlMagnitude = Math.sqrt(curl.reduce((sum, c) => sum + c*c, 0));
                    
                    // Масштабирование
                    const vecMagnitude = Math.sqrt(vec.reduce((sum, v) => sum + v*v, 0));
                    const scale = 0.3 * Math.min(dx, dy) / (vecMagnitude + 0.001);
                    
                    field.push({
                        position: [x, y],
                        vector: [vec[0] * scale, vec[1] * scale],
                        curl: curl,
                        curlMagnitude: curlMagnitude,
                        color: this.getColorByCurl(curlMagnitude)
                    });
                }
            }
            
            return field;
        }
        
        getColorByCurl(magnitude) {
            const maxCurl = 3;
            const t = Math.min(magnitude / maxCurl, 1);
            return d3.interpolateRgb('#059669', '#f59e0b')(t);
        }
        
        // Вспомогательные функции
        calculateTriangleNormal(triangle) {
            const [p0, p1, p2] = triangle;
            const v1 = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];
            const v2 = [p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]];
            
            const normal = [
                v1[1]*v2[2] - v1[2]*v2[1],
                v1[2]*v2[0] - v1[0]*v2[2],
                v1[0]*v2[1] - v1[1]*v2[0]
            ];
            
            const length = Math.sqrt(normal.reduce((sum, n) => sum + n*n, 0));
            return normal.map(n => n / length);
        }
        
        calculateTriangleArea(triangle) {
            const [p0, p1, p2] = triangle;
            const a = Math.sqrt(this.distanceSquared(p0, p1));
            const b = Math.sqrt(this.distanceSquared(p1, p2));
            const c = Math.sqrt(this.distanceSquared(p2, p0));
            const s = (a + b + c) / 2;
            return Math.sqrt(s * (s - a) * (s - b) * (s - c));
        }
        
        distanceSquared(p1, p2) {
            return p1.reduce((sum, v, i) => sum + (v - p2[i])*(v - p2[i]), 0);
        }
        
        triangulateSurface(surface, resolution) {
            // Простая триангуляция для примера
            const triangles = [];
            
            if (surface.type === 'plane') {
                const [xMin, xMax, yMin, yMax, z] = surface.params;
                const dx = (xMax - xMin) / resolution;
                const dy = (yMax - yMin) / resolution;
                
                for (let i = 0; i < resolution; i++) {
                    for (let j = 0; j < resolution; j++) {
                        const x1 = xMin + i * dx;
                        const y1 = yMin + j * dy;
                        const x2 = x1 + dx;
                        const y2 = y1 + dy;
                        
                        // Два треугольника на каждый квадрат
                        triangles.push([
                            [x1, y1, z],
                            [x2, y1, z],
                            [x2, y2, z]
                        ]);
                        
                        triangles.push([
                            [x1, y1, z],
                            [x2, y2, z],
                            [x1, y2, z]
                        ]);
                    }
                }
            }
            
            return triangles;
        }
    }

    /**
     * Лапласиан и уравнения в частных производных
     */
    class Laplacian {
        constructor() {
            this.equations = {
                // Уравнение Лапласа: ∇²φ = 0
                laplace: (x, y) => {
                    // Фундаментальные решения
                    const r = Math.sqrt(x*x + y*y) + 0.001;
                    return Math.log(r); // 2D: ln(r)
                },
                
                // Уравнение Пуассона: ∇²φ = f
                poisson: (x, y) => {
                    // Решение для точечного источника
                    const r = Math.sqrt(x*x + y*y) + 0.001;
                    return -r*r * Math.log(r) / 4; // Решение для f = 1
                },
                
                // Уравнение Гельмгольца: ∇²φ + k²φ = 0
                helmholtz: (x, y, k = 1) => {
                    const r = Math.sqrt(x*x + y*y) + 0.001;
                    return Math.cos(k * r); // Приближенное решение
                },
                
                // Волновое уравнение: ∂²φ/∂t² = c²∇²φ
                wave: (x, t, c = 1) => {
                    return Math.sin(x - c*t) + Math.sin(x + c*t);
                },
                
                // Уравнение теплопроводности: ∂φ/∂t = α∇²φ
                heat: (x, t, alpha = 1) => {
                    return Math.exp(-alpha*t) * Math.sin(x);
                }
            };
        }
        
        // Численный лапласиан
        calculate(f, point, h = 0.0001) {
            const dim = point.length;
            let laplacian = 0;
            
            for (let i = 0; i < dim; i++) {
                const pointPlus = [...point];
                const pointMinus = [...point];
                
                pointPlus[i] += h;
                pointMinus[i] -= h;
                
                laplacian += (f(...pointPlus) - 2*f(...point) + f(...pointMinus)) / (h * h);
            }
            
            return laplacian;
        }
        
        // Решение методом конечных разностей
        solveFiniteDifference(equationType, bounds, boundaryConditions, resolution = 50) {
            const grid = this.createGrid(bounds, resolution);
            const solution = this.initializeSolution(grid, boundaryConditions);
            
            // Итерационный метод (метод Якоби)
            const maxIterations = 1000;
            const tolerance = 1e-6;
            
            for (let iter = 0; iter < maxIterations; iter++) {
                const newSolution = this.jacobiIteration(solution, grid, equationType);
                const error = this.calculateError(solution, newSolution);
                
                // Копируем новое решение
                for (let i = 0; i < resolution; i++) {
                    for (let j = 0; j < resolution; j++) {
                        solution[i][j] = newSolution[i][j];
                    }
                }
                
                if (error < tolerance) break;
            }
            
            return {
                grid: grid,
                solution: solution,
                bounds: bounds
            };
        }
        
        createGrid(bounds, resolution) {
            const [xMin, xMax, yMin, yMax] = bounds;
            const dx = (xMax - xMin) / (resolution - 1);
            const dy = (yMax - yMin) / (resolution - 1);
            
            const grid = [];
            for (let i = 0; i < resolution; i++) {
                const row = [];
                for (let j = 0; j < resolution; j++) {
                    row.push({
                        x: xMin + j * dx,
                        y: yMin + i * dy,
                        isBoundary: (i === 0 || i === resolution-1 || j === 0 || j === resolution-1)
                    });
                }
                grid.push(row);
            }
            
            return grid;
        }
        
        initializeSolution(grid, boundaryConditions) {
            const resolution = grid.length;
            const solution = Array(resolution).fill().map(() => Array(resolution).fill(0));
            
            // Применяем граничные условия
            for (let i = 0; i < resolution; i++) {
                for (let j = 0; j < resolution; j++) {
                    if (grid[i][j].isBoundary) {
                        const {x, y} = grid[i][j];
                        solution[i][j] = boundaryConditions(x, y);
                    }
                }
            }
            
            return solution;
        }
        
        jacobiIteration(solution, grid, equationType) {
            const resolution = grid.length;
            const newSolution = JSON.parse(JSON.stringify(solution));
            const [xMin, xMax, yMin, yMax] = [grid[0][0].x, grid[0][resolution-1].x, 
                                              grid[0][0].y, grid[resolution-1][0].y];
            const dx = (xMax - xMin) / (resolution - 1);
            const dy = (yMax - yMin) / (resolution - 1);
            
            for (let i = 1; i < resolution - 1; i++) {
                for (let j = 1; j < resolution - 1; j++) {
                    if (!grid[i][j].isBoundary) {
                        const {x, y} = grid[i][j];
                        
                        // Пятиточечный шаблон для лапласиана
                        const laplacian = (solution[i-1][j] - 2*solution[i][j] + solution[i+1][j])/(dy*dy) +
                                         (solution[i][j-1] - 2*solution[i][j] + solution[i][j+1])/(dx*dx);
                        
                        switch(equationType) {
                            case 'laplace':
                                newSolution[i][j] = 0.25 * (solution[i-1][j] + solution[i+1][j] + 
                                                           solution[i][j-1] + solution[i][j+1]);
                                break;
                            case 'poisson':
                                const f = 1; // Плотность источника
                                newSolution[i][j] = 0.25 * (solution[i-1][j] + solution[i+1][j] + 
                                                           solution[i][j-1] + solution[i][j+1] - dx*dy*f);
                                break;
                        }
                    }
                }
            }
            
            return newSolution;
        }
        
        calculateError(oldSol, newSol) {
            let maxError = 0;
            const resolution = oldSol.length;
            
            for (let i = 0; i < resolution; i++) {
                for (let j = 0; j < resolution; j++) {
                    const error = Math.abs(newSol[i][j] - oldSol[i][j]);
                    maxError = Math.max(maxError, error);
                }
            }
            
            return maxError;
        }
        
        // Визуализация решения
        createVisualizationData(solution, grid) {
            const data = {
                x: [],
                y: [],
                z: []
            };
            
            const resolution = grid.length;
            
            for (let i = 0; i < resolution; i++) {
                for (let j = 0; j < resolution; j++) {
                    data.x.push(grid[i][j].x);
                    data.y.push(grid[i][j].y);
                    data.z.push(solution[i][j]);
                }
            }
            
            return data;
        }
    }

    // Экспорт классов
    return {
        Gradient,
        Divergence,
        Curl,
        Laplacian,
        
        // Вспомогательные функции для 3D визуализации
        init3DScene: function(canvasId) {
            if (typeof THREE === 'undefined') {
                console.error('THREE.js не загружен');
                return null;
            }
            
            this.canvas3D = document.getElementById(canvasId);
            if (!this.canvas3D) return null;
            
            // Сцена
            this.scene = new THREE.Scene();
            this.scene.background = new THREE.Color(0x0f172a);
            
            // Камера
            this.camera = new THREE.PerspectiveCamera(
                75,
                this.canvas3D.clientWidth / this.canvas3D.clientHeight,
                0.1,
                1000
            );
            this.camera.position.z = 5;
            
            // Рендерер
            this.renderer = new THREE.WebGLRenderer({ 
                canvas: this.canvas3D,
                antialias: true 
            });
            this.renderer.setSize(this.canvas3D.clientWidth, this.canvas3D.clientHeight);
            
            // Освещение
            const ambientLight = new THREE.AmbientLight(0x404040);
            this.scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
            directionalLight.position.set(1, 1, 1);
            this.scene.add(directionalLight);
            
            // Орбитальный контрол
            if (typeof THREE.OrbitControls !== 'undefined') {
                this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
                this.controls.enableDamping = true;
                this.controls.dampingFactor = 0.05;
            }
            
            // Анимация
            const animate = () => {
                requestAnimationFrame(animate);
                if (this.controls) this.controls.update();
                this.renderer.render(this.scene, this.camera);
            };
            animate();
            
            // Обработка изменения размера
            window.addEventListener('resize', () => {
                this.camera.aspect = this.canvas3D.clientWidth / this.canvas3D.clientHeight;
                this.camera.updateProjectionMatrix();
                this.renderer.setSize(this.canvas3D.clientWidth, this.canvas3D.clientHeight);
            });
            
            return {
                scene: this.scene,
                camera: this.camera,
                renderer: this.renderer,
                controls: this.controls
            };
        },
        
        createVectorField3D: function(fieldFunction, bounds, resolution = 10) {
            if (!this.scene) return null;
            
            const [xMin, xMax, yMin, yMax, zMin, zMax] = bounds;
            const dx = (xMax - xMin) / resolution;
            const dy = (yMax - yMin) / resolution;
            const dz = (zMax - zMin) / resolution;
            
            const arrows = new THREE.Group();
            
            for (let i = 0; i <= resolution; i++) {
                for (let j = 0; j <= resolution; j++) {
                    for (let k = 0; k <= resolution; k++) {
                        const x = xMin + i * dx;
                        const y = yMin + j * dy;
                        const z = zMin + k * dz;
                        
                        const vector = fieldFunction(x, y, z);
                        const magnitude = Math.sqrt(vector[0]*vector[0] + 
                                                   vector[1]*vector[1] + 
                                                   vector[2]*vector[2]);
                        
                        if (magnitude > 0.001) {
                            const arrow = this.createArrow(x, y, z, vector, magnitude);
                            arrows.add(arrow);
                        }
                    }
                }
            }
            
            this.scene.add(arrows);
            return arrows;
        },
        
        createArrow: function(x, y, z, vector, magnitude) {
            const arrowLength = magnitude * 0.3;
            const arrowHelper = new THREE.ArrowHelper(
                new THREE.Vector3(...vector).normalize(),
                new THREE.Vector3(x, y, z),
                arrowLength,
                0xff0000,
                0.2 * arrowLength,
                0.1 * arrowLength
            );
            
            return arrowHelper;
        },
        
        createScalarField3D: function(scalarFunction, bounds, resolution = 20) {
            if (!this.scene) return null;
            
            const [xMin, xMax, yMin, yMax, zMin, zMax] = bounds;
            const dx = (xMax - xMin) / resolution;
            const dy = (yMax - yMin) / resolution;
            const dz = (zMax - zMin) / resolution;
            
            // Создаем геометрию для точек
            const geometry = new THREE.BufferGeometry();
            const vertices = [];
            const colors = [];
            
            // Находим мин и макс для цветовой шкалы
            let minVal = Infinity;
            let maxVal = -Infinity;
            
            for (let i = 0; i <= resolution; i++) {
                for (let j = 0; j <= resolution; j++) {
                    for (let k = 0; k <= resolution; k++) {
                        const x = xMin + i * dx;
                        const y = yMin + j * dy;
                        const z = zMin + k * dz;
                        
                        const val = scalarFunction(x, y, z);
                        minVal = Math.min(minVal, val);
                        maxVal = Math.max(maxVal, val);
                    }
                }
            }
            
            // Создаем точки
            for (let i = 0; i <= resolution; i++) {
                for (let j = 0; j <= resolution; j++) {
                    for (let k = 0; k <= resolution; k++) {
                        const x = xMin + i * dx;
                        const y = yMin + j * dy;
                        const z = zMin + k * dz;
                        
                        const val = scalarFunction(x, y, z);
                        const t = (val - minVal) / (maxVal - minVal);
                        
                        vertices.push(x, y, z);
                        
                        // Цвет от синего к красному
                        const color = new THREE.Color();
                        color.setHSL(0.7 * (1 - t), 0.9, 0.5);
                        colors.push(color.r, color.g, color.b);
                    }
                }
            }
            
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            
            const material = new THREE.PointsMaterial({
                size: 0.05,
                vertexColors: true,
                transparent: true,
                opacity: 0.7
            });
            
            const points = new THREE.Points(geometry, material);
            this.scene.add(points);
            
            return points;
        }
    };
}

// Создаем глобальный экземпляр
if (typeof window !== 'undefined') {
    window.VectorAnalysis = new VectorAnalysis();
}

export default VectorAnalysis;