# frozen_string_literal: true

# OpenStudio Measure Template
# Reference scaffold for idiomatic measure development.
# Replace class name, arguments, and logic as needed.

require 'openstudio'
require 'openstudio/measure/ShowRunnerOutput'

class ExampleMeasure < OpenStudio::Measure::ModelMeasure
  def name
    'Example Measure'
  end

  def description
    'One-line description of what this measure does and when to apply it.'
  end

  def modeler_description
    'Technical description of the algorithm, assumptions, and any ASHRAE references.'
  end

  def arguments(model)
    args = OpenStudio::Measure::OSArgumentVector.new

    # Example: string argument with default
    arg = OpenStudio::Measure::OSArgument.makeStringArgument('example_arg', true)
    arg.setDisplayName('Example Argument')
    arg.setDescription('Description of what this controls and its valid range.')
    arg.setDefaultValue('default_value')
    args << arg

    # Example: double argument with bounds
    arg = OpenStudio::Measure::OSArgument.makeDoubleArgument('efficiency', true)
    arg.setDisplayName('Equipment Efficiency')
    arg.setDescription('Rated efficiency (0.0-1.0). Per ASHRAE 90.1 Table 6.8.1.')
    arg.setDefaultValue(0.85)
    args << arg

    # Example: choice argument
    ch = OpenStudio::StringVector.new
    ch << 'OptionA'
    ch << 'OptionB'
    arg = OpenStudio::Measure::OSArgument.makeChoiceArgument('climate_zone', ch, true)
    arg.setDisplayName('Climate Zone')
    arg.setDescription('ASHRAE climate zone designation.')
    arg.setDefaultValue('OptionA')
    args << arg

    # Example: bool argument
    arg = OpenStudio::Measure::OSArgument.makeBoolArgument('apply_sizing', true)
    arg.setDisplayName('Apply Sizing')
    arg.setDescription('Whether to trigger a sizing run after modifications.')
    arg.setDefaultValue(true)
    args << arg

    args
  end

  def run(model, runner, user_arguments)
    super(model, runner, user_arguments)

    # Validate arguments
    unless runner.validateUserArguments(arguments(model), user_arguments)
      return false
    end

    example_arg = runner.getStringArgumentValue('example_arg', user_arguments)
    efficiency = runner.getDoubleArgumentValue('efficiency', user_arguments)
    climate_zone = runner.getStringArgumentValue('climate_zone', user_arguments)
    apply_sizing = runner.getBoolArgumentValue('apply_sizing', user_arguments)

    # --- Physical validation ---
    # Validate inputs against physical constraints BEFORE modifying the model.
    # Fail fast with a clear message if assumptions are violated.
    unless efficiency > 0.0 && efficiency <= 1.0
      runner.registerError("Efficiency #{efficiency} is outside physical bounds (0.0, 1.0].")
      return false
    end

    runner.registerInitialCondition("Model has #{model.getSpaces.size} spaces.")

    # --- Model modifications ---
    # Apply changes here. Register info/Warning/Error as appropriate.
    # Example:
    # model.getElectricEquipments.each do |equip|
    #   old_eff = equip.electricEquipmentDefinition.designLevel.get
    #   runner.registerInfo("Updating #{equip.name}: #{old_eff} -> #{efficiency}")
    # end

    runner.registerInfo("Applied changes with efficiency=#{efficiency}, climate_zone=#{climate_zone}.")

    # --- Post-modification ---
    if apply_sizing
      runner.registerInfo('Note: sizing run required. Handle in workflow or via runner.haventImplementedYet.')
    end

    runner.registerFinalCondition("Measure applied. Modified model has #{model.getSpaces.size} spaces.")
    true
  end
end

# Register with OpenStudio
ExampleMeasure.new.registerWithApplication
