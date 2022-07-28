'use strict';
let cardColor = config.colors.white;
let headingColor = config.colors.headingColor;
let axisColor = config.colors.axisColor;
let borderColor = config.colors.borderColor;
let shadeColor;

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
	apiKey: "AIzaSyAUwC9sGdHEhNJZxTX3BJL_7L2HU0FzbmM",
	authDomain: "pollution-fcc8d.firebaseapp.com",
	databaseURL: "https://pollution-fcc8d-default-rtdb.firebaseio.com",
	projectId: "pollution-fcc8d",
	storageBucket: "pollution-fcc8d.appspot.com",
	messagingSenderId: "750667679514",
	appId: "1:750667679514:web:bfb3d06a066eed326f4ea3",
	measurementId: "G-2N1WPMPYFX"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);

/**
 * Moves the map to display over Cape Town
 *
 * @param  {H.Map} map      A HERE Map instance within the application
 */
function moveMapToCapTown(map) {
	map.setCenter({ lat: -33.918861, lng: 18.423300 });
	map.setZoom(12);
}

/**
 * Get data from the Cloud and display the points on the map
 *
 * @param  {H.Map} map      					A HERE Map instance within the application
 */
function displayNodes(map) {
	firebase.database().ref('nodes').on('value', (snapshot) => { 
		var nodes = snapshot.val();
		for (let node in nodes) {
			for (let info in nodes[node]) {
				let location = nodes[node][info]['location'];
				map.addObject(new H.map.Marker(location));
			}
		}
	});
}

/**
 * Get data from the Cloud and display the points on the map
 *
 */
function displayStatsData() {
	let name = document.getElementById("name").textContent;
	let device_id = document.getElementById("device_id").textContent;

	firebase.database().ref('data').on('value', (snapshot) => {
		var nodes = snapshot.val();
		var dataseries = [];
		var predictions = null;

		for (let value in nodes[name]) {
			if (nodes[name][value]['device'].localeCompare(device_id) == true) {
				dataseries.push([nodes[name][value]['timestamp'] * 1000, nodes[name][value]['decibels']['ave']]);
				predictions = nodes[name][value];
			}
		}

		// Income Chart - Area chart
		// --------------------------------------------------------------------
		const incomeChartEl = document.querySelector('#incomeChart');
		let	incomeChartConfig = {
				series: [
				{
						name: "Decibels",
						data: dataseries
					}
				],
				chart: {
					height: 215,
					parentHeightOffset: 0,
					parentWidthOffset: 0,
					toolbar: {
						show: false
					},
					type: 'area',
					animations: {
						initialAnimation: {
							enabled: true
						}
					}
				},
				dataLabels: {
					enabled: false
				},
				stroke: {
					width: 2,
					curve: 'smooth'
				},
				legend: {
					show: false
				},
				markers: {
					size: 6,
					colors: 'transparent',
					strokeColors: 'transparent',
					strokeWidth: 4,
					discrete: [
						{
							fillColor: config.colors.white,
							seriesIndex: 0,
							dataPointIndex: 7,
							strokeColor: config.colors.primary,
							strokeWidth: 2,
							size: 6,
							radius: 8
						}
					],
					hover: {
						size: 7
					}
				},
				colors: [config.colors.primary],
				fill: {
					type: 'gradient',
					gradient: {
						shade: shadeColor,
						shadeIntensity: 0.6,
						opacityFrom: 0.5,
						opacityTo: 0.25,
						stops: [0, 95, 100]
					}
				},
				grid: {
					borderColor: borderColor,
					strokeDashArray: 3,
					padding: {
						top: -20,
						bottom: -8,
						left: -10,
						right: 8
					}
				},
				xaxis: {
					type: 'datetime',
					axisBorder: {
						show: false
					},
					axisTicks: {
						show: false
					},
					labels: {
						show: true,
						style: {
							fontSize: '13px',
							colors: axisColor
						}
					}
				},
			yaxis: {
					labels: {
						show: false
					},
					min: -100,
					max: 0,
					tickAmount: 4
				}
			};
		
		if (typeof incomeChartEl !== undefined && incomeChartEl !== null) {
			const incomeChart = new ApexCharts(incomeChartEl, incomeChartConfig);
			incomeChart.render();
		}

		// Expenses Mini Chart - Radial Chart
		// --------------------------------------------------------------------
		const weeklyExpensesEl = document.querySelector('#expensesOfWeek');
		if (predictions != null) {
			let conf = Math.floor(predictions['confidence'] * 100);
			let weeklyExpensesConfig = {
				series: [conf],
				chart: {
					width: 60,
					height: 60,
					type: 'radialBar'
				},
				plotOptions: {
					radialBar: {
						startAngle: 0,
						endAngle: 360,
						strokeWidth: '8',
						hollow: {
							margin: 2,
							size: '50%'
						},
						track: {
							strokeWidth: '50%',
							background: borderColor
						},
						dataLabels: {
							show: true,
							name: {
								show: false
							},
							value: {
								formatter: function (val) {
									return '' + parseInt(val) + '%';
								},
								offsetY: 5,
								color: '#697a8d',
								fontSize: '13px',
								show: true
							}
						}
					}
				},
				fill: {
					type: 'solid',
					colors: config.colors.primary
				},
				stroke: {
					lineCap: 'round'
				},
				grid: {
					padding: {
						top: -10,
						bottom: -15,
						left: -10,
						right: -10
					}
				},
				states: {
					hover: {
						filter: {
							type: 'none'
						}
					},
					active: {
						filter: {
							type: 'none'
						}
					}
				}
			};
		
		
			if (typeof weeklyExpensesEl !== undefined && weeklyExpensesEl !== null) {
				const weeklyExpenses = new ApexCharts(weeklyExpensesEl, weeklyExpensesConfig);
				weeklyExpenses.render();
			
				let xtime = new Date(predictions['timestamp'] * 1000);
				let xtime_sample = "" + xtime.getDate() + "/" + (xtime.getMonth() + 1) + "/" + xtime.getFullYear() + " " + xtime.getHours() +
					":" + xtime.getMinutes() +
					":" + xtime.getSeconds();
			
				document.getElementById("predictions").innerHTML = predictions['audio'];
				document.getElementById("datetime").innerHTML = xtime_sample;
			}
		}
	});
}

/**
 * Boilerplate map initialization code starts below:
 */

//Step 1: initialize communication with the platform
// In your own code, replace variable window.apikey with your own apikey
var platform = new H.service.Platform({
	apikey: '5Y_viqWR1ra1hdbL5tDiFNgXc6m4v5T6jCnybuy5noY'
});
var defaultLayers = platform.createDefaultLayers();

//Step 2: initialize a map - this map is centered over Europe
var map = new H.Map(document.getElementById('map'),
	defaultLayers.vector.normal.map, {
		center: { lat: -33.918861, lng: 8.423300 },
	zoom: 4,
	pixelRatio: window.devicePixelRatio || 1
});
// add a resize listener to make sure that the map occupies the whole container
window.addEventListener('resize', () => map.getViewPort().resize());

//Step 3: make the map interactive
// MapEvents enables the event system
// Behavior implements default interactions for pan/zoom (also on mobile touch environments)
var behavior = new H.mapevents.Behavior(new H.mapevents.MapEvents(map));

// Create the default UI components
var ui = H.ui.UI.createDefault(map, defaultLayers);

// Now use the map as required...
window.onload = function () {
	moveMapToCapTown(map);
	displayNodes(map);
	displayStatsData();
}

