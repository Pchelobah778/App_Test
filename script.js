document.addEventListener('DOMContentLoaded', function() {
    // Конфигурация
    const config = {
        width: 1200,
        height: 700,
        nodeClickAnimationDuration: 300,
        tooltipDelay: 200
    };

    // Инициализация SVG
    const svg = d3.select('#tree-svg')
        .attr('width', config.width)
        .attr('height', config.height);

    // Создание контейнеров
    const defs = svg.append('defs');
    const linkGroup = svg.append('g').attr('class', 'links');
    const nodeGroup = svg.append('g').attr('class', 'nodes');
    const labelGroup = svg.append('g').attr('class', 'labels');

    // Загрузка данных
    d3.json('data.json').then(data => {
        // Подготавливаем данные
        prepareData(data);
        
        // Создаем градиенты
        createGradients(defs, data.nodes);
        
        // Рисуем связи
        drawLinks(data);
        
        // Рисуем узлы
        drawNodes(data);
        
        // Добавляем подписи
        drawLabels(data);
        
        // Настраиваем взаимодействие
        setupInteractions(data);
        
        // Настраиваем масштабирование
        setupZoom();
        
    }).catch(error => {
        console.error('Error loading data:', error);
        alert('Ошибка загрузки данных. Проверьте файл data.json');
    });

    function prepareData(data) {
        // Создаем карту узлов по id для быстрого доступа
        data.nodeMap = {};
        data.nodes.forEach(node => {
            data.nodeMap[node.id] = node;
        });
        
        // Добавляем информацию о связях для каждого узла
        data.nodes.forEach(node => {
            node.links = data.links.filter(link => 
                link.source === node.id || link.target === node.id
            );
        });
    }

    function createGradients(defs, nodes) {
        nodes.filter(node => node.color && node.color.type).forEach(node => {
            let gradient;
            
            if (node.color.type === 'linear') {
                gradient = defs.append('linearGradient')
                    .attr('id', `gradient-${node.id}`)
                    .attr('x1', '0%')
                    .attr('y1', '0%')
                    .attr('x2', '100%')
                    .attr('y2', '100%');
            } else {
                gradient = defs.append('radialGradient')
                    .attr('id', `gradient-${node.id}`)
                    .attr('cx', '50%')
                    .attr('cy', '50%')
                    .attr('r', '50%')
                    .attr('fx', '50%')
                    .attr('fy', '50%');
            }
            
            node.color.stops.forEach(stop => {
                gradient.append('stop')
                    .attr('offset', stop.offset)
                    .attr('stop-color', stop.color);
            });
        });
    }

    function drawLinks(data) {
        // Функция для вычисления пути связи с кривизной
        function linkPath(d) {
            const source = typeof d.source === 'object' ? d.source : data.nodeMap[d.source];
            const target = typeof d.target === 'object' ? d.target : data.nodeMap[d.target];
            
            if (!source || !target) return '';
            
            const startX = source.x;
            const startY = source.y;
            const endX = target.x;
            const endY = target.y;
            
            // Вычисляем контрольные точки для кривой Безье
            const curvature = d.curvature || 0;
            const dx = endX - startX;
            const dy = endY - startY;
            const midX = (startX + endX) / 2;
            const midY = (startY + endY) / 2;
            
            // Перпендикулярный вектор для кривизны
            const perpX = -dy * curvature;
            const perpY = dx * curvature;
            
            const controlX = midX + perpX;
            const controlY = midY + perpY;
            
            return `M${startX},${startY} Q${controlX},${controlY} ${endX},${endY}`;
        }
        
        // Рисуем связи
        linkGroup.selectAll('.link')
            .data(data.links)
            .enter()
            .append('path')
            .attr('class', 'link')
            .attr('d', linkPath)
            .attr('stroke', d => d.color || '#95a5a6')
            .attr('stroke-width', d => d.width || 2)
            .attr('fill', 'none')
            .attr('opacity', d => d.opacity || 0.7)
            .attr('stroke-dasharray', d => d.dashed ? '5,5' : null);
    }

    function drawNodes(data) {
        // Функция для получения цвета узла
        function getNodeColor(node) {
            if (typeof node.color === 'string') {
                return node.color;
            } else if (node.color && node.color.type) {
                return `url(#gradient-${node.id})`;
            }
            return '#3498db';
        }
        
        // Рисуем узлы
        nodeGroup.selectAll('.node')
            .data(data.nodes)
            .enter()
            .append('circle')
            .attr('class', 'node')
            .attr('cx', d => d.x)
            .attr('cy', d => d.y)
            .attr('r', d => d.size || 10)
            .attr('fill', getNodeColor)
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .attr('data-id', d => d.id);
    }

    function drawLabels(data) {
        // Рисуем подписи
        labelGroup.selectAll('.node-label')
            .data(data.nodes)
            .enter()
            .append('text')
            .attr('class', 'node-label')
            .attr('x', d => d.x)
            .attr('y', d => d.y - d.size - 5)
            .attr('text-anchor', 'middle')
            .text(d => d.name);
    }

    function setupInteractions(data) {
        const tooltip = d3.select('#tooltip');
        const nodeInfo = d3.select('#node-info');
        let tooltipTimer;
        
        // Обработчики для узлов
        d3.selectAll('.node')
            .on('mouseover', function(event, d) {
                // Очищаем предыдущий таймер
                clearTimeout(tooltipTimer);
                
                // Подсветка узла
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('r', d.size * 1.2);
                
                // Подсветка связанных элементов
                highlightConnections(d.id, true, data);
                
                // Задержка перед показом тултипа
                tooltipTimer = setTimeout(() => {
                    tooltip
                        .style('opacity', 1)
                        .html(`
                            <strong>${d.name}</strong><br>
                            <em>${d.description || 'Нет описания'}</em><br>
                            <small>Кликните для перехода</small>
                        `)
                        .style('left', (event.pageX + 10) + 'px')
                        .style('top', (event.pageY - 10) + 'px');
                }, config.tooltipDelay);
            })
            .on('mouseout', function(event, d) {
                // Очищаем таймер тултипа
                clearTimeout(tooltipTimer);
                
                // Возврат к исходному размеру
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('r', d.size);
                
                // Снятие подсветки
                highlightConnections(d.id, false, data);
                
                // Скрытие тултипа
                tooltip.style('opacity', 0);
            })
            .on('click', function(event, d) {
                event.stopPropagation();
                
                // Анимация клика
                d3.select(this)
                    .transition()
                    .duration(config.nodeClickAnimationDuration / 2)
                    .attr('r', d.size * 1.5)
                    .transition()
                    .duration(config.nodeClickAnimationDuration / 2)
                    .attr('r', d.size);
                
                // Обновление информации в панели
                updateInfoPanel(d, nodeInfo);
                
                // Переход по ссылке (если она не ссылается на текущую страницу)
                if (d.url && d.url !== '#') {
                    setTimeout(() => {
                        window.location.href = d.url;
                    }, config.nodeClickAnimationDuration * 2);
                }
            });
        
        // Обработчики для связей
        d3.selectAll('.link')
            .on('mouseover', function(event, d) {
                // Подсветка связи
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('stroke-width', (d.width || 2) * 1.5)
                    .attr('opacity', 1);
                
                // Находим связанные узлы
                const source = typeof d.source === 'object' ? d.source : data.nodeMap[d.source];
                const target = typeof d.target === 'object' ? d.target : data.nodeMap[d.target];
                
                if (source) highlightNode(source.id, true);
                if (target) highlightNode(target.id, true);
            })
            .on('mouseout', function(event, d) {
                // Возврат связи к исходному виду
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('stroke-width', d.width || 2)
                    .attr('opacity', d.opacity || 0.7);
                
                // Снятие подсветки узлов
                const source = typeof d.source === 'object' ? d.source : data.nodeMap[d.source];
                const target = typeof d.target === 'object' ? d.target : data.nodeMap[d.target];
                
                if (source) highlightNode(source.id, false);
                if (target) highlightNode(target.id, false);
            });
        
        // Обработчики кнопок управления
        d3.select('#reset-view').on('click', () => {
            svg.transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity);
        });
        
        d3.select('#toggle-labels').on('click', () => {
            const labels = d3.selectAll('.node-label');
            const isVisible = labels.style('opacity') !== '0';
            labels
                .transition()
                .duration(300)
                .style('opacity', isVisible ? 0 : 1);
        });
        
        // Клик по пустому месту сбрасывает выделение
        svg.on('click', function(event) {
            if (event.target === this) {
                nodeInfo.html('<p>Кликните на узел для получения информации</p>');
                resetAllHighlights(data);
            }
        });
    }
    
    function updateInfoPanel(node, nodeInfo) {
        nodeInfo.html(`
            <h4>${node.name}</h4>
            <p class="description">${node.description || 'Описание отсутствует'}</p>
            <div class="node-details">
                <p><strong>Координаты:</strong> (${Math.round(node.x)}, ${Math.round(node.y)})</p>
                <p><strong>Размер узла:</strong> ${node.size}</p>
                <p><strong>Количество связей:</strong> ${node.links.length}</p>
            </div>
            ${node.url && node.url !== '#' ? `
                <a href="${node.url}" class="node-link" target="_blank">
                    Перейти к разделу →
                </a>
            ` : ''}
        `);
    }
    
    function highlightNode(nodeId, highlight) {
        const node = d3.select(`.node[data-id="${nodeId}"]`);
        if (!node.empty()) {
            node
                .transition()
                .duration(200)
                .attr('r', d => highlight ? d.size * 1.3 : d.size)
                .attr('filter', highlight ? 'url(#glow)' : null);
        }
    }
    
    function highlightConnections(nodeId, highlight, data) {
        const node = data.nodeMap[nodeId];
        if (!node) return;
        
        // Подсвечиваем связанные связи
        d3.selectAll('.link')
            .attr('opacity', function(link) {
                const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                
                const isConnected = sourceId === nodeId || targetId === nodeId;
                return highlight ? (isConnected ? 1 : 0.2) : (link.opacity || 0.7);
            })
            .attr('stroke-width', function(link) {
                const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                
                const isConnected = sourceId === nodeId || targetId === nodeId;
                return highlight ? (isConnected ? (link.width || 2) * 1.5 : link.width || 2) : (link.width || 2);
            });
        
        // Подсвечиваем связанные узлы
        d3.selectAll('.node')
            .attr('opacity', function(d) {
                if (d.id === nodeId) return 1;
                
                const isConnected = node.links.some(link => 
                    link.source === d.id || link.target === d.id
                );
                return highlight ? (isConnected ? 1 : 0.3) : 1;
            });
    }
    
    function resetAllHighlights(data) {
        d3.selectAll('.link')
            .attr('opacity', d => d.opacity || 0.7)
            .attr('stroke-width', d => d.width || 2);
        
        d3.selectAll('.node')
            .attr('opacity', 1)
            .attr('filter', null);
    }
    
    function setupZoom() {
        // Создаем эффект свечения для подсветки
        const defs = svg.select('defs');
        const filter = defs.append('filter')
            .attr('id', 'glow')
            .attr('height', '300%')
            .attr('width', '300%')
            .attr('x', '-100%')
            .attr('y', '-100%');
            
        filter.append('feGaussianBlur')
            .attr('stdDeviation', '3.5')
            .attr('result', 'coloredBlur');
            
        const feMerge = filter.append('feMerge');
        feMerge.append('feMergeNode')
            .attr('in', 'coloredBlur');
        feMerge.append('feMergeNode')
            .attr('in', 'SourceGraphic');
        
        // Настраиваем масштабирование
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                svg.selectAll('g')
                    .attr('transform', event.transform);
            });
        
        svg.call(zoom);
    }
});