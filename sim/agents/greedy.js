// Greedy VGC agent: scores every legal action with a damage estimate plus a few support-move rules.
//
// By default it reads the live Battle object, so it knows the opponent's true stats, abilities and
// moves. With `blind: true` it plays under closed-team rules: your own team in full, but for the
// opponent only species (hence types and base stats), visible HP percentage, boosts and status.
// Unknown opponent stats are guessed from base stats, and an opposing Pokémon is assumed to carry
// a 80-power attack of each of its types.
//
// Everything is tunable in the SCORES table at the top.

const path = require('path');
const SHOWDOWN = path.join(__dirname, '..', '..', 'vendor', 'pokemon-showdown', 'dist', 'sim');
const {RandomPlayerAI} = require(path.join(SHOWDOWN, 'tools', 'random-player-ai'));

const SCORES = {
	koBonus: 0.6,         // extra value for finishing a target
	priorityKoBonus: 0.3, // KO with a priority move is safer
	allyHitPenalty: 1.2,  // multiplier on damage a spread move does to our own partner
	protect: 0.3,
	fakeOut: 0.5,
	speedControl: 0.9,    // Trick Room / Tailwind when it helps
	screen: 0.4,
	helpingHand: 0.45,
	redirect: 0.5,
	heal: 0.9,
	statusMove: 0.4,
	setupMove: 0.35,
	fieldMove: 0.25,
	otherStatus: 0.05,
	roleFakeOut: 0.8,
	roleSpeedControl: 0.6,
	roleWeather: 0.5,
	roleRedirect: 0.4,
	roleIntimidate: 0.4,
	megaBring: 1.5,
};

const PROTECT_MOVES = new Set(['protect', 'detect', 'kingsshield', 'spikyshield', 'banefulbunker',
	'obstruct', 'silktrap', 'burningbulwark']);
const HEAL_MOVES = new Set(['recover', 'roost', 'softboiled', 'slackoff', 'milkdrink', 'synthesis',
	'morningsun', 'moonlight', 'shoreup', 'healorder']);
const REDIRECT_MOVES = new Set(['followme', 'ragepowder']);
const FIELD_MOVES = {raindance: 'raindance', sunnyday: 'sunnyday', sandstorm: 'sandstorm',
	snowscape: 'snowscape', chillyreception: 'snowscape'};
const TERRAIN_MOVES = new Set(['electricterrain', 'grassyterrain', 'mistyterrain', 'psychicterrain']);
const SETUP_BOOST = new Set(['swordsdance', 'calmmind', 'nastyplot', 'dragondance', 'quiverdance',
	'bulkup', 'irondefense', 'agility', 'shellsmash']);
const WEATHER_ABILITIES = new Set(['drought', 'drizzle', 'sandstream', 'snowwarning',
	'orichalcumpulse', 'electricsurge', 'grassysurge', 'psychicsurge', 'mistysurge', 'hadronengine']);
// Abilities that make a type's moves useless against the holder.
const ABILITY_IMMUNITY = {levitate: 'Ground', voltabsorb: 'Electric', lightningrod: 'Electric',
	motordrive: 'Electric', waterabsorb: 'Water', stormdrain: 'Water', dryskin: 'Water',
	flashfire: 'Fire', sapsipper: 'Grass'};

const alive = mon => mon && !mon.fainted && mon.hp > 0;

// Effective stat from stored stats, boosts and the common stat items. Deliberately avoids
// Pokemon.getStat(), which fires ability events that expect a move in progress.
function stat(mon, name) {
	const boost = mon.boosts[name] || 0;
	let value = mon.storedStats[name] * (boost >= 0 ? (2 + boost) / 2 : 2 / (2 - boost));
	if (mon.item === 'choicescarf' && name === 'spe') value *= 1.5;
	if (mon.item === 'choiceband' && name === 'atk') value *= 1.5;
	if (mon.item === 'choicespecs' && name === 'spa') value *= 1.5;
	if (mon.item === 'assaultvest' && name === 'spd') value *= 1.5;
	return value;
}

const GUESS_BONUS = 36, GUESS_HP_BONUS = 91; // a mid-level stat-point investment for unknown stats

class GreedyPlayer extends RandomPlayerAI {
	constructor(stream, options, getBattle) {
		super(stream, {...options, mega: 1});
		this.getBattle = getBattle;
		this.blind = !!(options && options.blind);
	}

	// Does the closed-team rule hide this Pokémon's real stats from us?
	hidden(mon) { return this.blind && mon.side !== this.me; }

	statOf(mon, name) {
		if (!this.hidden(mon)) return stat(mon, name);
		const boost = mon.boosts[name] || 0;
		const guess = mon.species.baseStats[name] + GUESS_BONUS;
		return guess * (boost >= 0 ? (2 + boost) / 2 : 2 / (2 - boost));
	}

	hpOf(mon) {
		if (!this.hidden(mon)) return mon.hp;
		return (mon.hp / mon.maxhp) * (mon.species.baseStats.hp + GUESS_HP_BONUS);
	}

	receiveRequest(request) {
		this.request = request;
		this.claimed = [0, 0]; // damage already planned on each foe slot this turn (fraction of its HP)
		super.receiveRequest(request);
	}

	get battle() { return this.getBattle(); }
	get me() { return this.battle.sides[this.request.side.id === 'p1' ? 0 : 1]; }
	get foe() { return this.battle.sides[this.request.side.id === 'p1' ? 1 : 0]; }

	// ---- damage estimate -------------------------------------------------------------------

	// Expected damage of `move` from `attacker` on `defender`, as a fraction of defender's current HP.
	estimate(attacker, defender, move) {
		if (move.category === 'Status' || !alive(defender)) return 0;
		const dex = this.battle.dex;
		if (!dex.getImmunity(move.type, defender)) return 0;
		if (!this.hidden(defender) && ABILITY_IMMUNITY[defender.ability] === move.type) return 0;

		let power = move.basePower || 60; // variable-power moves: assume a middling 60
		if (move.multihit) power *= Array.isArray(move.multihit) ? 3.1 : move.multihit;
		const physical = move.category === 'Physical';
		const attack = this.statOf(attacker, move.id === 'bodypress' ? 'def' : physical ? 'atk' : 'spa');
		const defense = this.statOf(defender, physical ? 'def' : 'spd');
		let damage = Math.floor(Math.floor(22 * power * attack / defense) / 50) + 2;

		if (attacker.hasType(move.type)) damage *= attacker.ability === 'adaptability' ? 2 : 1.5;
		damage *= 2 ** dex.getEffectiveness(move.type, defender);
		if (move.target === 'allAdjacentFoes' || move.target === 'allAdjacent') damage *= 0.75;
		if (attacker.status === 'brn' && physical && attacker.ability !== 'guts') damage *= 0.5;
		if (!this.hidden(attacker) && attacker.item === 'lifeorb') damage *= 1.3;

		const weather = this.battle.field.effectiveWeather();
		if (weather === 'sunnyday' || weather === 'desolateland') {
			if (move.type === 'Fire') damage *= 1.5;
			if (move.type === 'Water') damage *= 0.5;
		} else if (weather === 'raindance' || weather === 'primordialsea') {
			if (move.type === 'Water') damage *= 1.5;
			if (move.type === 'Fire') damage *= 0.5;
		}
		damage *= 0.925; // average of the random roll
		return damage / this.hpOf(defender);
	}

	bestDamage(attacker, defender) {
		let best = 0;
		if (this.hidden(attacker)) {
			// Their moves are unknown: assume a strong attack of each of their types.
			const physical = attacker.species.baseStats.atk >= attacker.species.baseStats.spa;
			for (const type of attacker.getTypes()) {
				const threat = {id: 'threat', type, category: physical ? 'Physical' : 'Special', basePower: 80, target: 'normal'};
				best = Math.max(best, this.estimate(attacker, defender, threat));
			}
			return best;
		}
		for (const slot of attacker.moveSlots) {
			best = Math.max(best, this.estimate(attacker, defender, this.battle.dex.moves.get(slot.id)));
		}
		return best;
	}

	// ---- move choice -----------------------------------------------------------------------

	chooseMove(active, moves) {
		const options = this.rankMoves(this.request.active.indexOf(active), moves);
		const best = options[0];
		this.recordPlan(this.me.active[this.request.active.indexOf(active)], best);
		return best.choice;
	}

	// Every legal move-and-target for the Pokémon in `slot`, best score first.
	rankMoves(slot, moves) {
		const mon = this.me.active[slot];
		const ally = this.me.active[slot ^ 1];
		const options = [];

		for (const m of moves) {
			const move = this.battle.dex.moves.get(m.move.move);
			const base = `move ${m.move.slot}`;
			const target = m.move.target;
			if (['normal', 'any', 'adjacentFoe'].includes(target)) {
				const foeSlots = [1, 2].filter(t => alive(this.foe.active[t - 1]));
				for (const t of foeSlots.length ? foeSlots : [1]) {
					options.push({choice: `${base} ${t}`, score: this.scoreMove(mon, ally, move, t), move, foeSlot: t});
				}
			} else if (target === 'adjacentAlly') {
				options.push({choice: `${base} -${(slot ^ 1) + 1}`, score: this.scoreMove(mon, ally, move, null), move});
			} else if (target === 'adjacentAllyOrSelf') {
				options.push({choice: `${base} -${slot + 1}`, score: this.scoreMove(mon, ally, move, null), move});
			} else {
				options.push({choice: base, score: this.scoreMove(mon, ally, move, null), move});
			}
		}

		options.sort((a, b) => b.score - a.score);
		return options;
	}

	// Remember planned damage so the partner doesn't also spend its attack on a target already dying.
	recordPlan(mon, best) {
		const {move, foeSlot} = best;
		if (move.category === 'Status') return;
		if (foeSlot) {
			this.claimed[foeSlot - 1] += this.estimate(mon, this.foe.active[foeSlot - 1], move);
		} else if (move.target === 'allAdjacentFoes' || move.target === 'allAdjacent') {
			this.foe.active.forEach((f, i) => { this.claimed[i] += this.estimate(mon, f, move); });
		}
	}

	scoreMove(mon, ally, move, foeSlot) {
		const accuracy = move.accuracy === true ? 1 : (move.accuracy || 100) / 100;

		if (move.category !== 'Status') {
			let score = 0;
			const foes = foeSlot ? [foeSlot] : [1, 2];
			const spread = move.target === 'allAdjacentFoes' || move.target === 'allAdjacent';
			if (!foeSlot && !spread) return 0; // self-targeting damage moves are not worth modelling
			for (const t of foes) {
				const target = this.foe.active[t - 1];
				if (!alive(target)) continue;
				const damage = this.estimate(mon, target, move);
				const remaining = Math.max(0, 1 - this.claimed[t - 1]);
				score += Math.min(damage, remaining);
				if (damage >= remaining && remaining > 0) {
					score += SCORES.koBonus + (move.priority > 0 ? SCORES.priorityKoBonus : 0);
				}
			}
			if (move.target === 'allAdjacent' && alive(ally)) {
				score -= SCORES.allyHitPenalty * Math.min(1, this.estimate(mon, ally, move));
			}
			if (move.id === 'fakeout' || move.id === 'firstimpression') {
				if (mon.activeTurns > 0) return -1; // only works on the first turn out
				const target = this.foe.active[foeSlot - 1];
				if (target && this.battle.dex.getImmunity('Normal', target)) score += SCORES.fakeOut;
			}
			if (move.recoil) score -= 0.1;
			return score * accuracy;
		}
		return this.scoreStatus(mon, ally, move, foeSlot);
	}

	scoreStatus(mon, ally, move, foeSlot) {
		const id = move.id;
		const hp = mon.hp / mon.maxhp;
		const field = this.battle.field;
		const target = foeSlot ? this.foe.active[foeSlot - 1] : null;

		if (PROTECT_MOVES.has(id)) {
			return mon.lastMove && PROTECT_MOVES.has(mon.lastMove.id) ? -1 : SCORES.protect;
		}
		if (id === 'trickroom') {
			const mine = this.me.active.filter(alive).map(p => this.statOf(p, 'spe'));
			const theirs = this.foe.active.filter(alive).map(p => this.statOf(p, 'spe'));
			const avg = a => a.reduce((x, y) => x + y, 0) / (a.length || 1);
			const wantsRoom = avg(mine) < avg(theirs);
			const active = !!field.pseudoWeather.trickroom;
			return wantsRoom !== active ? SCORES.speedControl : -0.2;
		}
		if (id === 'tailwind') {
			return this.me.sideConditions.tailwind || field.pseudoWeather.trickroom ? -0.5 : SCORES.speedControl * 0.9;
		}
		if (['reflect', 'lightscreen', 'auroraveil'].includes(id)) {
			if (this.me.sideConditions[id]) return -0.5;
			if (id === 'auroraveil' && !['snowscape', 'snow', 'hail'].includes(field.effectiveWeather())) return -1;
			return SCORES.screen;
		}
		if (id === 'helpinghand') return alive(ally) ? SCORES.helpingHand : -1;
		if (REDIRECT_MOVES.has(id)) return alive(ally) ? SCORES.redirect : -0.5;
		if (HEAL_MOVES.has(id) || (move.heal && move.target === 'self')) {
			return hp < 0.5 ? SCORES.heal * (1 - hp) * 2 : -0.3;
		}
		if (FIELD_MOVES[id]) return field.effectiveWeather() === FIELD_MOVES[id] ? -0.5 : SCORES.fieldMove;
		if (TERRAIN_MOVES.has(id)) return field.terrain === id ? -0.5 : SCORES.fieldMove;
		if (SETUP_BOOST.has(id)) return hp > 0.6 && (mon.boosts.atk + mon.boosts.spa) < 2 ? SCORES.setupMove : -0.2;

		if (move.status && target) {
			const status = target.status;
			if (status || !alive(target)) return -0.5;
			if (move.flags.powder && target.hasType('Grass')) return -0.5;
			if (move.status === 'par' && target.hasType('Electric')) return -0.5;
			if (move.status === 'brn' && target.hasType('Fire')) return -0.5;
			if ((move.status === 'psn' || move.status === 'tox') &&
				(target.hasType('Poison') || target.hasType('Steel'))) return -0.5;
			const bonus = move.status === 'slp' ? 0.3 : 0;
			return (SCORES.statusMove + bonus) * ((move.accuracy === true ? 100 : move.accuracy || 100) / 100);
		}
		return SCORES.otherStatus;
	}

	// ---- switching -------------------------------------------------------------------------

	chooseSwitch(active, switches) {
		const foes = this.foe.active.filter(alive);
		let best = switches[0].slot, bestScore = -Infinity;
		for (const {slot} of switches) {
			const candidate = this.me.pokemon[slot - 1];
			let score = candidate.hp / candidate.maxhp;
			for (const foe of foes) {
				score += Math.min(this.bestDamage(candidate, foe), 1.5) - 0.7 * Math.min(this.bestDamage(foe, candidate), 1.5);
			}
			if (score > bestScore) { bestScore = score; best = slot; }
		}
		return best;
	}

	// ---- team preview ----------------------------------------------------------------------

	chooseTeamPreview(team) {
		const dex = this.battle.dex;
		const mine = this.me.pokemon;
		const theirs = this.foe.pokemon;

		const rated = mine.map((mon, i) => {
			let score = 0;
			for (const foe of theirs) {
				score += Math.min(this.bestDamage(mon, foe), 1.5) - 0.6 * Math.min(this.bestDamage(foe, mon), 1.5);
			}
			const moveIds = mon.moveSlots.map(s => s.id);
			let lead = score;
			if (moveIds.includes('fakeout')) { score += SCORES.roleFakeOut; lead += SCORES.roleFakeOut * 2; }
			if (moveIds.includes('tailwind') || moveIds.includes('trickroom')) {
				score += SCORES.roleSpeedControl;
				lead += moveIds.includes('tailwind') ? SCORES.roleSpeedControl : 0;
			}
			if (WEATHER_ABILITIES.has(mon.baseAbility)) { score += SCORES.roleWeather; lead += SCORES.roleWeather; }
			if (moveIds.some(m => REDIRECT_MOVES.has(m))) { score += SCORES.roleRedirect; lead += SCORES.roleRedirect; }
			if (mon.baseAbility === 'intimidate') { score += SCORES.roleIntimidate; lead += SCORES.roleIntimidate; }
			if (dex.items.get(mon.item).megaStone) score += SCORES.megaBring;
			return {index: i + 1, score, lead};
		});

		const bring = rated.sort((a, b) => b.score - a.score).slice(0, 4);
		bring.sort((a, b) => b.lead - a.lead);
		return `team ${bring.map(b => b.index).join('')}`;
	}
}

module.exports = {GreedyPlayer};
