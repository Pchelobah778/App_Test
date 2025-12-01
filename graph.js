// Конфигурация
let config = {};
let nodes = [];
let edges = [];
let zoom = d3.zoom();
let currentZoom = 1;

// Загрузка данных
async function loadData() {
    try {
        const response = await fetch('data.json');
        config = await response.json();
        nodes = config.nodes;
        edges = config.edges;
        
        initGraph();
    } catch (error) {
        console.error('Ошибка загрузки data.json:', error);
        document.getElementById('graph-container').innerHTML = 
            '<p class="error">Ошибка загрузки данных. Проверьте файл data.json</p>';
    }
}

// Инициализация графа
function initGraph() {
    const settings = config.settings || {};
    const canvas = settings.canvas || { width: 1200, height: 800 };
    const nodeDefaults = settings.nodeDefaults || {};
    const edgeDefaults = settings.edgeDefaults || {};
    
    // Очистка SVG
    d3.select('#graph-svg').selectAll('*').remove();
    
    // Создание SVG
    const svg = d3.select('#graph-svg')
        .attr('width', canvas.width)
        .attr('height', canvas.height)
        .style('background-color', canvas.backgroundColor || '#f8f9fa');
    
    // Добавление зума
    const g = svg.append('g');
    
    zoom = d3.zoom()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
            currentZoom = event.transform.k;
            document.getElementById('zoom-level').textContent = 
                `${Math.round(currentZoom * 100)}%`;
        });
    
    svg.call(zoom);
    
    // Рисуем ребра
    const edgesGroup = g.append('g').attr('class', 'edges');
    
    edges.forEach(edge => {
        const sourceNode = nodes.find(n => n.id === edge.source);
        const targetNode = nodes.find(n => n.id === edge.target);
        
        if (!sourceNode || !targetNode) return;
        
        const line = edgesGroup.append('g');
        
        // Рисуем линию
        if (edge.lineStyle === 'curved') {
            // Квадратичная кривая Безье
            const midX = (sourceNode.x + targetNode.x) / 2;
            const midY = (sourceNode.y + targetNode.y) / 2;
            const dx = targetNode.x - sourceNode.x;
            const dy = targetNode.y - sourceNode.y;
            const curvature = edge.curvature || 0;
            
            const cp1x = midX + curvature * dy;
            const cp1y = midY - curvature * dx;
            
            line.append('path')
                .attr('d', `M${sourceNode.x},${sourceNode.y} Q${cp1x},${cp1y} ${targetNode.x},${targetNode.y}`)
                .attr('fill', 'none')
                .attr('stroke', edge.color || edgeDefaults.color || '#858796')
                .attr('stroke-width', edge.width || edgeDefaults.width || 2)
                .attr('stroke-dasharray', edge.dashed ? (edge.dashArray || '5,5') : null)
                .attr('class', 'edge-line');
        } else {
            // Прямая линия
            line.append('line')
                .attr('x1', sourceNode.x)
                .attr('y1', sourceNode.y)
                .attr('x2', targetNode.x)
                .attr('y2', targetNode.y)
                .attr('stroke', edge.color || edgeDefaults.color || '#858796')
                .attr('stroke-width', edge.width || edgeDefaults.width || 2)
                .attr('stroke-dasharray', edge.dashed ? (edge.dashArray || '5,5') : null)
                .attr('class', 'edge-line');
        }
    });
    
    // Рисуем узлы - ВСЕ КВАДРАТНЫЕ
    const nodesGroup = g.append('g').attr('class', 'nodes');
    
    nodes.forEach(node => {
        const nodeGroup = nodesGroup.append('g')
            .attr('class', 'node')
            .attr('data-id', node.id)
            .attr('transform', `translate(${node.x},${node.y})`)
            .style('cursor', 'pointer')
            .on('click', () => {
                if (node.link) {
                    window.open(node.link, '_blank');
                }
            })
            .on('mouseover', function(event) {

            })
            .on('mouseout', function(event) {

            });
        
        const size = (node.radius || nodeDefaults.radius || 40) * 1.414; // Диагональ квадрата = сторона * √2
        const side = size / Math.sqrt(2); // Сторона квадрата
        
        nodeGroup.append('rect')
            .attr('class', 'node-shape')
            .attr('x', -side/2)
            .attr('y', -side/2)
            .attr('width', side)
            .attr('height', side)
            .attr('rx', node.borderRadius || 5) // Закругление углов (можно настроить)
            .attr('ry', node.borderRadius || 5)
            .attr('fill', node.color || nodeDefaults.color || '#4e73df')
            .attr('stroke', node.borderColor || nodeDefaults.borderColor || '#2e59d9')
            .attr('stroke-width', node.borderWidth || nodeDefaults.borderWidth || 2);
        
        // Добавляем текст
        nodeGroup.append('text')
            .attr('class', 'node-label')
            .attr('text-anchor', 'middle')
            .attr('dy', '0.3em')
            .attr('fill', node.textColor || nodeDefaults.textColor || '#ffffff')
            .attr('font-size', node.fontSize || nodeDefaults.fontSize || '14px')
            .attr('font-weight', 'bold')
            .text(node.label);
    });
}

// Вспомогательная функция для изменения цвета
function adjustColor(color, amount) {
    // Простой способ осветлить цвет
    return d3.color(color).brighter(amount/20);
}

// Генерация точек для шестиугольника (больше не используется, но оставлю на всякий случай)
function generateHexagonPoints(radius) {
    const points = [];
    for (let i = 0; i < 6; i++) {
        const angle = Math.PI / 3 * i;
        const x = radius * Math.cos(angle);
        const y = radius * Math.sin(angle);
        points.push(`${x},${y}`);
    }
    return points.join(' ');
}

// Показ информации об узле
function showNodeInfo(node) {
    const infoDiv = document.getElementById('node-info');
    infoDiv.innerHTML = `
        <strong>${node.label}</strong><br>
        ${node.link ? `Ссылка: <a href="${node.link}" target="_blank">${node.link}</a>` : 'Нет ссылки'}
    `;
}

// Управление зумом
function zoomIn() {
    d3.select('#graph-svg')
        .transition()
        .duration(300)
        .call(zoom.scaleBy, 1.2);
}

function zoomOut() {
    d3.select('#graph-svg')
        .transition()
        .duration(300)
        .call(zoom.scaleBy, 0.8);
}

function resetView() {
    d3.select('#graph-svg')
        .transition()
        .duration(300)
        .call(zoom.transform, d3.zoomIdentity);
}

// Загрузка данных при старте
document.addEventListener('DOMContentLoaded', loadData);