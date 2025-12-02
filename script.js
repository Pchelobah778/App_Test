document.addEventListener('DOMContentLoaded', function() {
    const config = {
        width: 1200,
        height: 700,
        nodeStrength: -300,
        linkStrength: 0.2,
        chargeDistance: 200,
        linkDistance: 100
    };

    const svg = d3.select('#tree-svg')
        .attr('width', config.width)
        .attr('height', config.height);

    const linkGroup = svg.append('g').attr('class', 'links');
    const nodeGroup = svg.append('g').attr('class', 'nodes');
    const labelGroup = svg.append('g').attr('class', 'labels');

    d3.json('data.json').then(data => {
        initializeTree(data);
        setupInteractions();
    }).catch(error => {
        console.error('Error loading data:', error);
        alert('Ошибка загрузки данных. Проверьте файл data.json');
    });

    function initializeTree(data) {
        const simulation = d3.forceSimulation(data.nodes)
            .force('link', d3.forceLink(data.links)
                .id(d => d.id)
                .distance(config.linkDistance)
                .strength(config.linkStrength))
            .force('charge', d3.forceManyBody()
                .strength(config.nodeStrength)
                .distanceMax(config.chargeDistance))
            .force('center', d3.forceCenter(config.width / 2, config.height / 2))
            .force('collision', d3.forceCollide().radius(d => d.size + 5));

        const links = linkGroup.selectAll('.link')
            .data(data.links)
            .enter()
            .append('path')
            .attr('class', 'link')
            .attr('stroke', d => d.color || '#95a5a6')
            .attr('stroke-width', d => d.width || 2)
            .attr('fill', 'none')
            .attr('opacity', 0.7);

        createGradients(svg, data.nodes.filter(node => 
            node.color && typeof node.color === 'object'
        ));

        const nodes = nodeGroup.selectAll('.node')
            .data(data.nodes)
            .enter()
            .append('circle')
            .attr('class', 'node')
            .attr('r', d => d.size || 10)
            .attr('fill', d => getNodeColor(d))
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .call(drag(simulation));

        const labels = labelGroup.selectAll('.node-label')
            .data(data.nodes)
            .enter()
            .append('text')
            .attr('class', 'node-label')
            .text(d => d.name)
            .attr('text-anchor', 'middle')
            .attr('dy', d => -d.size - 5);

        simulation.on('tick', () => {
            links.attr('d', d => {
                const source = typeof d.source === 'object' ? d.source : data.nodes.find(n => n.id === d.source);
                const target = typeof d.target === 'object' ? d.target : data.nodes.find(n => n.id === d.target);
                
                if (!source || !target) return '';
                
                const curvature = d.curvature || 0;
                const midX = (source.x + target.x) / 2;
                const midY = (source.y + target.y) / 2;
                const dx = target.x - source.x;
                const dy = target.y - source.y;
                const length = Math.sqrt(dx * dx + dy * dy);
                
                const controlX = midX - dy * curvature;
                const controlY = midY + dx * curvature;
                
                return `M${source.x},${source.y} Q${controlX},${controlY} ${target.x},${target.y}`;
            });

            nodes
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            labels
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });

        window.treeData = data;
        window.treeSimulation = simulation;
    }

    function createGradients(svg, nodes) {
        const defs = svg.append('defs');
        
        nodes.forEach((node, i) => {
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
                    .attr('r', '50%');
            }
            
            node.color.stops.forEach(stop => {
                gradient.append('stop')
                    .attr('offset', stop.offset)
                    .attr('stop-color', stop.color);
            });
        });
    }

    function getNodeColor(node) {
        if (typeof node.color === 'string') {
            return node.color;
        } else if (node.color && node.color.type) {
            return `url(#gradient-${node.id})`;
        }
        return '#3498db';
    }

    function drag(simulation) {
        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }

        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }

        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }

        return d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended);
    }

    function setupInteractions() {
        const tooltip = d3.select('#tooltip');
        const nodeInfo = d3.select('#node-info');
        
        d3.selectAll('.node')
            .on('mouseover', function(event, d) {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('r', d.size * 1.2);

                highlightConnections(d.id, true);

                tooltip
                    .style('opacity', 1)
                    .html(`
                        <strong>${d.name}</strong><br>
                        <em>${d.description || 'Нет описания'}</em><br>
                        Размер: ${d.size}<br>
                        <small>Кликните для перехода</small>
                    `)
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 10) + 'px');
            })
            .on('mouseout', function(event, d) {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('r', d.size);

                highlightConnections(d.id, false);

                tooltip.style('opacity', 0);
            })
            .on('click', function(event, d) {
                event.stopPropagation();
                
                nodeInfo.html(`
                    <h4>${d.name}</h4>
                    <p>${d.description || 'Описание отсутствует'}</p>
                    <p><strong>Координаты:</strong> (${Math.round(d.x)}, ${Math.round(d.y)})</p>
                    <p><strong>Размер узла:</strong> ${d.size}</p>
                    ${d.url ? `<p><a href="${d.url}" class="node-link">Перейти к разделу →</a></p>` : ''}
                `);
                
                d3.select(this)
                    .transition()
                    .duration(300)
                    .attr('r', d.size * 1.5)
                    .transition()
                    .duration(300)
                    .attr('r', d.size);
                
                if (d.url) {
                    setTimeout(() => {
                        window.location.href = d.url;
                    }, 1000);
                }
            });

        d3.select('#reset-view').on('click', () => {
            if (window.treeSimulation) {
                window.treeSimulation.alpha(1).restart();
            }
        });

        d3.select('#toggle-labels').on('click', () => {
            const labels = d3.selectAll('.node-label');
            const isVisible = labels.style('display') !== 'none';
            labels.style('display', isVisible ? 'none' : 'block');
        });

        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                svg.selectAll('g')
                    .attr('transform', event.transform);
            });

        svg.call(zoom);
    }

    function highlightConnections(nodeId, highlight) {
        if (!window.treeData) return;

        const relatedLinks = window.treeData.links.filter(link => 
            link.source.id === nodeId || link.target.id === nodeId ||
            link.source === nodeId || link.target === nodeId
        );

        d3.selectAll('.link')
            .attr('opacity', d => {
                const isRelated = relatedLinks.includes(d);
                return highlight ? (isRelated ? 1 : 0.2) : 0.7;
            });

        d3.selectAll('.node')
            .attr('opacity', d => {
                const isRelated = d.id === nodeId || relatedLinks.some(link => 
                    link.source.id === d.id || link.target.id === d.id ||
                    link.source === d.id || link.target === d.id
                );
                return highlight ? (isRelated ? 1 : 0.3) : 1;
            });
    }
});