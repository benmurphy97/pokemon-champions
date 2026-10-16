// Plays battles for src/champions/arena.py. Reads jobs on stdin, writes results on stdout,
// and asks the Python side for every decision made by a "neural" player.
//
// One JSON object per line.
//   Python -> worker: {t:"job", id, a, b, agentA, agentB, seed}      teams are Showdown text
//                     {t:"act", key, choice}                         reply to a decision
//   worker -> Python: {t:"decide", key, id, side, lines, request}    lines = log lines since last decision
//                     {t:"result", id, winner: "A"|"B"|"tie", turns, fallbacks}
//
// Agents: neural (decisions from Python; team preview from the greedy agent so that every
// agent gets the same preview logic), greedy, random.

const path = require('path');
const readline = require('readline');
const {BattleStream, getPlayerStreams, Teams} = require(path.join(__dirname, '..', 'vendor', 'pokemon-showdown', 'dist', 'sim'));
const {RandomPlayerAI} = require(path.join(__dirname, '..', 'vendor', 'pokemon-showdown', 'dist', 'sim', 'tools', 'random-player-ai'));
const {GreedyPlayer} = require('./agents/greedy');

const FORMAT = 'gen9championsvgc2026regmc';
const pending = new Map(); // decision key -> resolve
let nextKey = 0;

function send(message) {
	process.stdout.write(JSON.stringify(message) + '\n');
}

function ask(id, side, lines, request) {
	const key = String(nextKey++);
	return new Promise(resolve => {
		pending.set(key, resolve);
		send({t: 'decide', key, id, side, lines, request});
	});
}

class NeuralPlayer extends GreedyPlayer {
	constructor(stream, options, getBattle, id, side) {
		super(stream, options, getBattle);
		this.id = id;
		this.side = side;
		this.lines = [];
		this.fallbacks = 0;
	}

	receiveLine(line) {
		if (line.startsWith('|') && !line.startsWith('|request|')) this.lines.push(line);
		super.receiveLine(line);
	}

	receiveRequest(request) {
		if (request.wait) return;
		if (request.teamPreview) return super.receiveRequest(request); // greedy team preview
		const lines = this.lines;
		this.lines = [];
		ask(this.id, this.side, lines, request).then(choice => this.choose(choice));
	}

	// An invalid choice from Python must not stall the battle: fall back to the engine default.
	receiveError(error) {
		this.fallbacks++;
		this.choose('default');
	}
}

function makeAgent(kind, stream, seed, getBattle, id, side) {
	if (kind === 'neural') return new NeuralPlayer(stream, {seed}, getBattle, id, side);
	if (kind === 'greedy') return new GreedyPlayer(stream, {seed}, getBattle);
	return new RandomPlayerAI(stream, {seed});
}

async function play(job) {
	const battleStream = new BattleStream();
	const streams = getPlayerStreams(battleStream);
	const getBattle = () => battleStream.battle;
	const seed = [job.seed, job.seed + 1, job.seed + 2, job.seed + 3];
	const p1 = makeAgent(job.agentA, streams.p1, seed.map(n => n + 100), getBattle, job.id, 'p1');
	const p2 = makeAgent(job.agentB, streams.p2, seed.map(n => n + 200), getBattle, job.id, 'p2');
	void p1.start();
	void p2.start();
	void streams.omniscient.write(
		`>start ${JSON.stringify({formatid: FORMAT, seed})}\n` +
		`>player p1 ${JSON.stringify({name: 'A', team: Teams.pack(Teams.import(job.a))})}\n` +
		`>player p2 ${JSON.stringify({name: 'B', team: Teams.pack(Teams.import(job.b))})}`);

	let winner = null, turns = 0;
	for await (const chunk of streams.omniscient) {
		for (const line of chunk.split('\n')) {
			if (line.startsWith('|turn|')) turns = Number(line.slice(6));
			else if (line.startsWith('|win|')) winner = line.slice(5) === 'A' ? 'A' : 'B';
			else if (line === '|tie') winner = 'tie';
		}
	}
	send({t: 'result', id: job.id, winner, turns, fallbacks: (p1.fallbacks || 0) + (p2.fallbacks || 0)});
}

const input = readline.createInterface({input: process.stdin});
let running = 0;
input.on('line', line => {
	const message = JSON.parse(line);
	if (message.t === 'act') {
		const resolve = pending.get(message.key);
		pending.delete(message.key);
		resolve(message.choice);
	} else if (message.t === 'job') {
		running++;
		play(message).catch(err => send({t: 'error', id: message.id, message: String(err.stack || err)}))
			.finally(() => { running--; });
	}
});
input.on('close', () => { if (running === 0) process.exit(0); });
